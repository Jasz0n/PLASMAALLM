"""Input types for the independent evaluator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvaluationInput(BaseModel):
    """Everything the evaluator reads; it never modifies these sources."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    topics: tuple[str, ...] = ()
    heldout_first: float | None = None
    heldout_last: float | None = None
    heldout_peak: float | None = None
    kel_lg: float | None = None
    kel_ks: float | None = None
    kel_cd: float | None = None
    kel_cre: float | None = None
    debate_disagreement: float | None = None
    review_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_forgetting_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    curriculum_score: float | None = Field(default=None, ge=0.0, le=1.0)
    alignment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    missing_knowledge: float | None = None
    conflict_discovery: float | None = None
