"""Attach Researcher multimodal evidence to unresolved debates."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from allm.debate.evidence import resolve_debate_evidence
from allm.debate.types import DebateResult
from allm.researcher.evidence_broker import EvidenceBroker


class DebateEvidenceSummary(BaseModel):
    """Evidence retrieved for one loop debate."""

    model_config = ConfigDict(frozen=True)

    query: str
    topic: str
    found: bool
    confidence: float = 0.0
    hit_count: int = 0
    summary: str = ""


def resolve_loop_debate_evidence(
    broker: EvidenceBroker,
    result: DebateResult,
    *,
    query: str | None = None,
) -> DebateEvidenceSummary:
    """Resolve multimodal evidence for an unresolved debate."""
    resolution = resolve_debate_evidence(broker, result, query=query)
    hits = resolution.bundle.hits
    return DebateEvidenceSummary(
        query=resolution.query,
        topic=resolution.topic,
        found=bool(hits),
        confidence=resolution.bundle.confidence,
        hit_count=len(hits),
        summary=resolution.bundle.summary,
    )
