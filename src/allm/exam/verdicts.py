"""Multi-dimensional exam verdicts (M44)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AlignmentLevel = Literal[
    "aligned",
    "partially_aligned",
    "disputed",
    "unsupported",
    "unknown",
]

_ALIGNMENT_SCORES: dict[str, float] = {
    "aligned": 1.0,
    "partially_aligned": 0.6,
    "disputed": 0.3,
    "unsupported": 0.1,
    "unknown": 0.5,
}


class MultiDimensionalVerdict(BaseModel):
    """Structured grading across curriculum, alignment, and evidence."""

    model_config = ConfigDict(frozen=True)

    curriculum_correct: bool
    curriculum_score: float = Field(ge=0.0, le=1.0)
    curriculum_reason: str | None = None
    alignment: AlignmentLevel = "unknown"
    alignment_score: float = Field(default=0.5, ge=0.0, le=1.0)
    alignment_reason: str | None = None
    evidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_reason: str | None = None
    exact_match: bool | None = None

    @classmethod
    def alignment_to_score(cls, level: AlignmentLevel) -> float:
        return _ALIGNMENT_SCORES.get(level, 0.5)


def mean_curriculum_score(results: tuple) -> float | None:
    """Mean curriculum score from question results carrying verdicts."""
    scores: list[float] = []
    for result in results:
        verdict = getattr(result, "verdict", None)
        if verdict is not None:
            scores.append(verdict.curriculum_score)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def mean_alignment_score(results: tuple) -> float | None:
    scores: list[float] = []
    for result in results:
        verdict = getattr(result, "verdict", None)
        if verdict is not None:
            scores.append(verdict.alignment_score)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def mean_evidence_score(results: tuple) -> float | None:
    scores: list[float] = []
    for result in results:
        verdict = getattr(result, "verdict", None)
        if verdict is not None:
            scores.append(verdict.evidence_score)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)
