"""Teacher-mediated 'show me' evidence retrieval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from allm.researcher.evidence_broker import EvidenceBroker, EvidenceBundle

SHOW_ME_KEYWORDS: tuple[str, ...] = (
    "plasma",
    "magnet",
    "field",
    "reactor",
    "motion",
    "stable",
    "unstable",
    "demonstration",
    "video",
)


class ConsultationEvidence(BaseModel):
    """Visual evidence attached to a mediated consultation."""

    model_config = ConfigDict(frozen=True)

    query: str
    found: bool
    confidence: float = 0.0
    hit_count: int = 0
    summary: str = ""


class ShowMeResult(BaseModel):
    """Outcome when Teacher requests visual evidence for students."""

    model_config = ConfigDict(frozen=True)

    asker_id: str
    topic: str
    query: str
    found: bool
    bundle: EvidenceBundle
    teacher_note: str = ""


def derive_show_me_query(prompt: str, topic: str) -> str:
    """Pick a retrieval query from a consultation prompt and topic."""
    lowered = prompt.lower()
    for keyword in SHOW_ME_KEYWORDS:
        if keyword in lowered:
            return keyword
    cleaned = topic.replace("-", " ").strip()
    return cleaned or "evidence"


def teacher_show_me(
    broker: EvidenceBroker,
    *,
    asker_id: str,
    topic: str,
    query: str,
    limit: int = 5,
) -> ShowMeResult:
    """Teacher retrieves synchronized evidence — students never touch raw video."""
    bundle = broker.show_me(query, topic=topic, limit=limit)
    found = bool(bundle.hits)
    note = bundle.summary if found else "No visual evidence available for this query."
    return ShowMeResult(
        asker_id=asker_id,
        topic=topic,
        query=query,
        found=found,
        bundle=bundle,
        teacher_note=note,
    )


def consultation_show_me(
    broker: EvidenceBroker,
    *,
    asker_id: str,
    topic: str,
    prompt: str,
    query: str | None = None,
    limit: int = 3,
) -> ConsultationEvidence:
    """Teacher retrieves evidence when a student asks 'show me' during consult."""
    resolved_query = query or derive_show_me_query(prompt, topic)
    result = teacher_show_me(
        broker,
        asker_id=asker_id,
        topic=topic,
        query=resolved_query,
        limit=limit,
    )
    return ConsultationEvidence(
        query=resolved_query,
        found=result.found,
        confidence=result.bundle.confidence,
        hit_count=len(result.bundle.hits),
        summary=result.teacher_note,
    )
