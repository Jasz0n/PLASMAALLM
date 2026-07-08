"""Debate evidence resolution — attach Researcher visuals to disagreements."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from allm.debate.types import DebateResult
from allm.researcher.evidence_broker import EvidenceBroker, EvidenceBundle


class DebateEvidenceResolution(BaseModel):
    """Evidence retrieved when a debate asks 'show me'."""

    model_config = ConfigDict(frozen=True)

    debate_question: str
    topic: str
    query: str
    bundle: EvidenceBundle
    unresolved: bool


def _default_query(result: DebateResult) -> str:
    """Derive a retrieval query from the debated question."""
    prompt = result.question.prompt.lower()
    for needle in ("plasma", "magnet", "field", "reactor", "motion"):
        if needle in prompt:
            return needle
    return result.question.topic or prompt.split()[0] if prompt else "evidence"


def resolve_debate_evidence(
    broker: EvidenceBroker,
    result: DebateResult,
    *,
    query: str | None = None,
    limit: int = 3,
) -> DebateEvidenceResolution:
    """Teacher requests evidence for an unresolved or challenged debate."""
    resolved_query = query or _default_query(result)
    bundle = broker.show_me(
        resolved_query,
        topic=result.question.topic,
        limit=limit,
    )
    return DebateEvidenceResolution(
        debate_question=result.question.prompt,
        topic=result.question.topic,
        query=resolved_query,
        bundle=bundle,
        unresolved=result.unresolved,
    )
