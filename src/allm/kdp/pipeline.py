"""The KDP compiler: Stages 1-8 orchestration + Stage 9 graph injection.

``KDPipeline.distill`` is a pure function of the ingested documents
(and the alias table): running it twice yields identical knowledge
units, ids included. ``GraphInjector`` then writes the result into the
Phase 5 knowledge graph — append-only, versioned, provenance preserved,
per both KDP.md section 9 and the graph's own invariants.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.kdp.cleaning import clean_document
from allm.kdp.transcript_cleaning import clean_transcript_document, looks_like_workshop_transcript
from allm.kdp.extraction import extract
from allm.kdp.ingestion import DocumentStore
from allm.kdp.merging import merge
from allm.kdp.embeddings import (
    ConceptEmbedder,
    DEFAULT_EMBED_THRESHOLD,
    embedding_threshold,
    resolve_embedder,
    resolve_embedding_config,
)
from allm.kdp.normalization import normalize
from allm.kdp.segmentation import segment
from allm.kdp.types import ConflictNode, KnowledgeUnit
from allm.knowledge.graph import KnowledgeGraph
from allm.knowledge.types import Concept, Evidence
from allm.storage.base import RecordStore

logger = get_logger("kdp.pipeline")


class DistillationResult(BaseModel):
    """Everything one KDP run produced."""

    model_config = ConfigDict(frozen=True)

    units: tuple[KnowledgeUnit, ...]
    conflicts: tuple[ConflictNode, ...]
    documents: int
    segments: int
    raw_units: int  # concept mentions before normalisation/merge (KEL: RCR denominator)


class KDPipeline:
    """Runs Stages 2-8 over a document store."""

    def __init__(
        self,
        aliases: dict[str, str] | None = None,
        *,
        use_embeddings: bool | None = None,
        embedder: ConceptEmbedder | None = None,
        embed_threshold: float | None = None,
    ) -> None:
        self._aliases = dict(aliases or {})
        env_embedder, env_threshold = resolve_embedding_config()

        if use_embeddings is False:
            self._embedder = None
        elif embedder is not None:
            self._embedder = embedder
        elif use_embeddings is True:
            backend = os.environ.get("ALLM_KDP_EMBED_BACKEND", "auto")
            self._embedder = resolve_embedder(backend)
        else:
            self._embedder = env_embedder

        if embed_threshold is not None:
            self._embed_threshold = embed_threshold
        elif self._embedder is not None:
            self._embed_threshold = float(
                os.environ.get("ALLM_KDP_EMBED_THRESHOLD", embedding_threshold(self._embedder))
            )
        else:
            self._embed_threshold = env_threshold or DEFAULT_EMBED_THRESHOLD

    def distill(self, store: DocumentStore) -> DistillationResult:
        documents = store.documents()
        cleaned: list = []
        for doc in documents:
            if looks_like_workshop_transcript(doc.text):
                cleaned.extend(clean_transcript_document(doc))
            else:
                cleaned.extend(clean_document(doc))
        segments = segment(cleaned)                                          # Stage 3
        raw_units = extract(segments)                                        # Stage 4
        normalized = normalize(
            raw_units,
            self._aliases,
            embedder=self._embedder,
            embed_threshold=self._embed_threshold,
        )                                                                      # Stage 5
        units, conflicts = merge(normalized)                                 # Stages 6-8
        logger.info(
            "distilled %d document(s) -> %d segment(s) -> %d unit(s), %d conflict(s)",
            len(documents),
            len(segments),
            len(units),
            len(conflicts),
        )
        return DistillationResult(
            units=tuple(units),
            conflicts=tuple(conflicts),
            documents=len(documents),
            segments=len(segments),
            raw_units=len(raw_units),
        )


class GraphInjector:
    """Stage 9 — writes distillation results into the knowledge graph.

    - New concepts are added with KDP's confidence and full evidence.
    - Existing concepts are *revised*: evidence and relations are
      appended; confidence is left to the teacher's exam-driven process
      (KDP measures textual stability, not the student's mastery).
    - Conflicts are persisted versioned (namespace ``conflicts``) for
      the exam engine and debate to consume, and attached to the
      concept as non-supporting evidence.
    """

    def __init__(self, graph: KnowledgeGraph, store: RecordStore | None = None) -> None:
        self._graph = graph
        self._store = store

    def inject(self, result: DistillationResult) -> dict[str, int]:
        added = revised = 0
        conflicted = {c.concept for c in result.conflicts}
        for unit in result.units:
            evidence = [
                Evidence(
                    source=span.doc,
                    detail=f"{unit.type} @ chars {span.start}-{span.end}",
                )
                for span in unit.raw_span_refs
            ]
            if unit.normalized_concept in conflicted:
                evidence.append(
                    Evidence(
                        source="kdp",
                        detail="sources disagree on this concept (see conflicts)",
                        supports=False,
                    )
                )
            existing = self._graph.get(unit.normalized_concept)
            if existing is None:
                self._graph.add(
                    Concept(
                        name=unit.normalized_concept,
                        description=unit.content.split("\n")[0][:200],
                        related=unit.relations,
                        confidence=unit.confidence,
                        evidence=tuple(evidence),
                        source="kdp",
                    ),
                    reason=f"kdp ingestion from {', '.join(unit.sources)}",
                )
                added += 1
            else:
                self._graph.revise(
                    unit.normalized_concept,
                    reason=f"kdp ingestion from {', '.join(unit.sources)}",
                    add_related=unit.relations,
                    add_evidence=evidence,
                )
                revised += 1

        for index, conflict in enumerate(result.conflicts):
            if self._store is not None:
                self._store.put(
                    "conflicts",
                    f"{conflict.concept}#{index}",
                    conflict.model_dump(mode="json"),
                    reason="kdp contradiction detection",
                )
        logger.info(
            "injected %d new / %d revised concept(s), %d conflict(s) stored",
            added,
            revised,
            len(result.conflicts),
        )
        return {"added": added, "revised": revised, "conflicts": len(result.conflicts)}
