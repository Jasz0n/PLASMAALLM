"""Stages 6-8 — Deduplication, Conflict Detection, Finalization.

Stage 6: units with the same (normalized concept, type) merge into one
KnowledgeUnit. Merging is strictly additive: every distinct explanation
is kept as a perspective, every source and span is attached. KDP never
picks "the correct version" — it builds the most structurally complete
one (KDP.md section 6).

Stage 7: two definition perspectives of the same concept conflict when
they tell different stories (almost no shared vocabulary) — or when
they tell the *same* story with different numbers ("exceeds 80 percent"
vs "capped near 10 percent"), the classic experimental dispute.
Conflicts are outputs, not errors — exam material and debate fuel
downstream.

Stage 8: confidence = stability, not truth. Computed from source
frequency, perspective consistency, structural clarity, and a
contradiction penalty. The formula is fixed and documented so scores
are reproducible and explainable.
"""

from __future__ import annotations

import re

from allm.kdp.segmentation import content_tokens
from allm.kdp.types import (
    ConflictNode,
    KnowledgeUnit,
    RawKnowledgeUnit,
    content_hash,
)

PERSPECTIVE_SEPARATOR = "\n\n--- perspective ---\n\n"
CONFLICT_OVERLAP_THRESHOLD = 0.15

# Confidence weights (must sum to 1).
_W_FREQUENCY = 0.4
_W_CONSISTENCY = 0.3
_W_CLARITY = 0.3
_CONFLICT_PENALTY = 0.5
_FULL_MARKS_SOURCES = 3


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _overlap(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _contradict(a: str, b: str) -> bool:
    """Two perspectives contradict when they tell different stories
    (near-zero shared vocabulary) or the same story with disjoint
    numeric claims (both quantify, no number in common)."""
    if _overlap(a, b) <= CONFLICT_OVERLAP_THRESHOLD:
        return True
    numbers_a, numbers_b = set(_NUMBER.findall(a)), set(_NUMBER.findall(b))
    return bool(numbers_a) and bool(numbers_b) and not (numbers_a & numbers_b)


def merge(units: list[RawKnowledgeUnit]) -> tuple[list[KnowledgeUnit], list[ConflictNode]]:
    """Deduplicate, detect conflicts, finalise. Deterministic ordering."""
    grouped: dict[tuple[str, str], list[RawKnowledgeUnit]] = {}
    for unit in units:
        grouped.setdefault((unit.concept, unit.type), []).append(unit)

    conflicts = _detect_conflicts(grouped)
    conflicted = {c.concept for c in conflicts}

    knowledge_units = []
    for (concept, kind), members in sorted(grouped.items()):
        perspectives = _unique_contents(members)
        sources = tuple(sorted({m.span.doc for m in members}))
        spans = tuple(
            sorted({m.span for m in members}, key=lambda s: (s.doc, s.start))
        )
        content = PERSPECTIVE_SEPARATOR.join(perspectives)
        knowledge_units.append(
            KnowledgeUnit(
                id=f"ku_{_slug(concept)}_{content_hash(concept, kind, content)}",
                type=kind,
                content=content,
                normalized_concept=concept,
                confidence=_confidence(perspectives, sources, concept in conflicted),
                sources=sources,
                context=_dominant_context(members),
                tags=tuple(sorted({t for m in members for t in m.tags})),
                raw_span_refs=spans,
            )
        )
    knowledge_units = _link_relations(knowledge_units)
    return knowledge_units, conflicts


def _detect_conflicts(
    grouped: dict[tuple[str, str], list[RawKnowledgeUnit]]
) -> list[ConflictNode]:
    """Divergent definitions of the same concept become ConflictNodes."""
    conflicts = []
    for (concept, kind), members in sorted(grouped.items()):
        if kind != "concept":
            continue
        perspectives = _unique_contents(members)
        for i, a in enumerate(perspectives):
            for b in perspectives[i + 1 :]:
                if _contradict(_definition_body(a, concept), _definition_body(b, concept)):
                    conflicts.append(
                        ConflictNode(
                            concept=concept,
                            interpretation_a=a,
                            interpretation_b=b,
                            sources=tuple(sorted({m.span.doc for m in members})),
                            evidence=tuple(
                                sorted({m.span for m in members}, key=lambda s: (s.doc, s.start))
                            ),
                        )
                    )
    return conflicts


def _definition_body(content: str, concept: str) -> str:
    """Compare what is *said about* the concept, not the concept words."""
    body = content
    for token in concept.split():
        body = body.replace(token, " ").replace(token.lower(), " ")
    return body


def _confidence(perspectives: list[str], sources: tuple[str, ...], in_conflict: bool) -> float:
    frequency = min(1.0, len(sources) / _FULL_MARKS_SOURCES)
    if len(perspectives) < 2:
        consistency = 1.0
    else:
        pairs = [
            _overlap(a, b)
            for i, a in enumerate(perspectives)
            for b in perspectives[i + 1 :]
        ]
        consistency = sum(pairs) / len(pairs)
    tokens = len(content_tokens(" ".join(perspectives)))
    clarity = min(1.0, tokens / 12)  # a defensible floor for "structured"
    score = _W_FREQUENCY * frequency + _W_CONSISTENCY * consistency + _W_CLARITY * clarity
    if in_conflict:
        score *= _CONFLICT_PENALTY
    return round(score, 4)


def _link_relations(units: list[KnowledgeUnit]) -> list[KnowledgeUnit]:
    """A unit relates to every other concept its content mentions."""
    names = {u.normalized_concept for u in units}
    linked = []
    for unit in units:
        mentioned = tuple(
            sorted(
                other
                for other in names
                if other != unit.normalized_concept
                and other.lower() in unit.content.lower()
            )
        )
        linked.append(unit.model_copy(update={"relations": mentioned}))
    return linked


def _unique_contents(members: list[RawKnowledgeUnit]) -> list[str]:
    seen: dict[str, str] = {}
    for member in sorted(members, key=lambda m: (m.span.doc, m.span.start)):
        key = " ".join(sorted(content_tokens(member.content)))
        seen.setdefault(key, member.content)
    return list(seen.values())


def _dominant_context(members: list[RawKnowledgeUnit]) -> str:
    counts: dict[str, int] = {}
    for member in members:
        counts[member.context] = counts.get(member.context, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _slug(concept: str) -> str:
    return "_".join(content_tokens(concept)) or "unknown"
