"""Evidence ledger: append-only package storage + the graph binder.

The ledger persists packages versioned (namespace ``evidence_packages``)
and answers "what evidence exists for concept X". The binder is the
write path into the knowledge graph: every submitted package becomes a
graph evidence entry on its concept, and the concept's confidence is
recomputed from *all* of its packages via the replication-aware
calculator — never set by hand, never influenced by who submitted.
"""

from __future__ import annotations

from allm.core.logging import get_logger
from allm.evidence.confidence import evidential_confidence
from allm.evidence.types import ConfidenceBreakdown, EvidencePackage
from allm.knowledge.graph import KnowledgeGraph
from allm.knowledge.types import Concept, Evidence
from allm.storage.base import RecordStore

logger = get_logger("evidence.ledger")

NAMESPACE = "evidence_packages"


class EvidenceLedger:
    """Versioned store of evidence packages."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def submit(self, package: EvidencePackage) -> EvidencePackage:
        self._store.put(
            NAMESPACE,
            package.id,
            package.model_dump(mode="json"),
            reason=f"{package.kind} by {package.contributor}: {package.outcome}",
        )
        logger.info(
            "package %s (%s, %s) for %r by %s",
            package.id,
            package.kind,
            package.outcome,
            package.concept,
            package.contributor,
        )
        return package

    def get(self, package_id: str) -> EvidencePackage | None:
        record = self._store.get(NAMESPACE, package_id)
        return None if record is None else EvidencePackage.model_validate(record.value)

    def packages_for(self, concept: str) -> list[EvidencePackage]:
        """All packages targeting a concept, oldest first."""
        packages = [
            p for p in self.all_packages() if p.concept == concept
        ]
        packages.sort(key=lambda p: (p.submitted_at, p.id))
        return packages

    def all_packages(self) -> list[EvidencePackage]:
        return [
            EvidencePackage.model_validate(self._store.get(NAMESPACE, key).value)
            for key in self._store.keys(NAMESPACE)
        ]

    def confidence(self, concept: str) -> ConfidenceBreakdown | None:
        return evidential_confidence(concept, self.packages_for(concept))


class EvidenceBinder:
    """Applies packages to the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph, ledger: EvidenceLedger) -> None:
        self._graph = graph
        self._ledger = ledger

    def submit(self, package: EvidencePackage) -> ConfidenceBreakdown:
        """Record the package and update its concept's belief.

        Creates the concept if this is the first anyone has heard of it
        (source ``evidence``); otherwise revises it. Either way the
        package lands as a graph evidence entry and confidence is
        recomputed from the full ledger.
        """
        self._ledger.submit(package)
        breakdown = self._ledger.confidence(package.concept)
        entry = Evidence(
            source=package.id,
            detail=f"{package.kind} ({package.outcome}): {package.claim}",
            supports=package.outcome == "supported",
        )
        reason = (
            f"evidence package {package.id} ({package.outcome}); "
            f"confidence {breakdown.value:.2f} from {breakdown.contributors} "
            f"contributor(s), {breakdown.independent_replications} replication(s)"
        )
        if self._graph.get(package.concept) is None:
            self._graph.add(
                Concept(
                    name=package.concept,
                    description=package.claim,
                    confidence=breakdown.value,
                    related=package.related_concepts,
                    evidence=(entry,),
                    source="evidence",
                ),
                reason=reason,
            )
        else:
            self._graph.revise(
                package.concept,
                reason=reason,
                confidence=breakdown.value,
                add_related=package.related_concepts,
                add_evidence=[entry],
            )
        return breakdown

    def why(self, concept: str) -> str:
        """Human-readable provenance tree (smallVision.md's example)."""
        breakdown = self._ledger.confidence(concept)
        if breakdown is None:
            return f"{concept}: no evidence packages recorded"
        lines = [f"{concept}  (confidence {breakdown.value:.2f})", "Evidence"]
        packages = self._ledger.packages_for(concept)
        for i, package in enumerate(packages):
            connector = "└──" if i == len(packages) - 1 else "├──"
            marker = {"supported": "+", "challenged": "-", "inconclusive": "?"}[
                package.outcome
            ]
            lines.append(
                f"{connector} [{marker}] {package.kind} {package.id} "
                f"by {package.contributor}"
            )
        return "\n".join(lines)
