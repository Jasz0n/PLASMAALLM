"""Expert lookup: which student knows a topic best."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from allm.teacher.state import KnowledgeState


class ExpertRanking(BaseModel):
    """Students ranked by confidence on one topic."""

    model_config = ConfigDict(frozen=True)

    topic: str
    rankings: tuple[tuple[str, float], ...]


def rank_experts(
    state: KnowledgeState,
    topic: str,
    *,
    min_confidence: float = 0.0,
) -> ExpertRanking:
    """Return students ordered by measured confidence on ``topic``."""
    rows: list[tuple[str, float]] = []
    for student_id in state.students():
        confidence = state.confidence(student_id, topic)
        if confidence is not None and confidence >= min_confidence:
            rows.append((student_id, confidence))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return ExpertRanking(topic=topic, rankings=tuple(rows))


def best_expert(
    state: KnowledgeState,
    topic: str,
    *,
    min_confidence: float = 0.0,
) -> str | None:
    """Student id with highest confidence on ``topic``, or None."""
    ranking = rank_experts(state, topic, min_confidence=min_confidence)
    if not ranking.rankings:
        return None
    return ranking.rankings[0][0]
