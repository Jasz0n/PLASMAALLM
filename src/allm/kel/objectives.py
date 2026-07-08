"""Multi-objective KEL — competing objectives and compromise decisions (M43)."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.evaluator.independent import EvaluationSnapshot

SteeringMode = Literal["learn", "maintain", "repair", "halt"]


class ObjectiveWeights(BaseModel):
    """Relative importance of each objective in compromise scoring."""

    model_config = ConfigDict(frozen=True)

    learning: float = Field(default=0.20, ge=0.0)
    stability: float = Field(default=0.25, ge=0.0)
    retention: float = Field(default=0.20, ge=0.0)
    generalization: float = Field(default=0.15, ge=0.0)
    evidence_quality: float = Field(default=0.10, ge=0.0)
    review_efficiency: float = Field(default=0.05, ge=0.0)
    contradiction_health: float = Field(default=0.05, ge=0.0)


class CompromiseDecision(BaseModel):
    """Best trade-off across competing objectives."""

    model_config = ConfigDict(frozen=True)

    mode: SteeringMode
    compromise_score: float
    snapshot: EvaluationSnapshot
    reason: str


def multi_objective_kel_enabled() -> bool:
    return os.environ.get("ALLM_MULTI_OBJECTIVE_KEL", "1") == "1"


def objective_weights_from_env() -> ObjectiveWeights:
    return ObjectiveWeights(
        learning=float(os.environ.get("ALLM_MO_LEARNING", "0.20")),
        stability=float(os.environ.get("ALLM_MO_STABILITY", "0.25")),
        retention=float(os.environ.get("ALLM_MO_RETENTION", "0.20")),
        generalization=float(os.environ.get("ALLM_MO_GENERALIZATION", "0.15")),
        evidence_quality=float(os.environ.get("ALLM_MO_EVIDENCE", "0.10")),
        review_efficiency=float(os.environ.get("ALLM_MO_EFFICIENCY", "0.05")),
        contradiction_health=float(os.environ.get("ALLM_MO_CONTRADICTIONS", "0.05")),
    )


def compromise_score(
    snapshot: EvaluationSnapshot,
    weights: ObjectiveWeights | None = None,
) -> float:
    """Weighted scalarization across normalized objectives."""
    w = weights or objective_weights_from_env()
    components = {
        "learning": snapshot.learning,
        "stability": snapshot.stability,
        "retention": snapshot.retention,
        "generalization": snapshot.generalization,
        "evidence_quality": snapshot.evidence_quality,
        "review_efficiency": snapshot.review_efficiency,
        "contradiction_health": snapshot.contradiction_health,
    }
    total_weight = sum(getattr(w, name) for name in components)
    if total_weight <= 0:
        return round(sum(components.values()) / len(components), 4)
    score = sum(getattr(w, name) * value for name, value in components.items()) / total_weight
    return round(score, 4)


def compromise_decision(
    snapshot: EvaluationSnapshot,
    *,
    weights: ObjectiveWeights | None = None,
    stability_floor: float | None = None,
    learning_floor: float | None = None,
) -> CompromiseDecision:
    """Pick learn / maintain / repair / halt from objective trade-offs."""
    floor_stability = stability_floor
    if floor_stability is None:
        floor_stability = float(os.environ.get("ALLM_KS_ADVANCE_THRESHOLD", "0.70"))
    floor_learning = learning_floor
    if floor_learning is None:
        floor_learning = float(os.environ.get("ALLM_MO_LEARNING_FLOOR", "0.35"))

    score = compromise_score(snapshot, weights)
    stable_enough = snapshot.stability >= floor_stability
    learning_ok = snapshot.learning >= floor_learning
    retention_ok = snapshot.retention >= floor_stability * 0.85

    if stable_enough and learning_ok and retention_ok:
        mode: SteeringMode = "learn"
        reason = (
            f"compromise {score:.2f}: stability {snapshot.stability:.2f} "
            f"and learning {snapshot.learning:.2f} both healthy"
        )
    elif not stable_enough and snapshot.learning >= floor_learning:
        mode = "maintain"
        reason = (
            f"compromise {score:.2f}: stability {snapshot.stability:.2f} "
            f"below {floor_stability:.2f} — prioritize retention"
        )
    elif snapshot.stability < floor_stability * 0.5 and snapshot.learning < floor_learning:
        mode = "repair"
        reason = (
            f"compromise {score:.2f}: stability {snapshot.stability:.2f} "
            f"and learning {snapshot.learning:.2f} both low"
        )
    elif not learning_ok and stable_enough:
        mode = "learn"
        reason = (
            f"compromise {score:.2f}: stable but learning {snapshot.learning:.2f} "
            f"below {floor_learning:.2f}"
        )
    else:
        mode = "maintain"
        reason = f"compromise {score:.2f}: balanced maintenance"

    return CompromiseDecision(
        mode=mode,
        compromise_score=score,
        snapshot=snapshot,
        reason=reason,
    )
