"""Independent evaluator — measurement-only, separate from Teacher (M43)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from allm.evaluator.types import EvaluationInput


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


class EvaluationSnapshot(BaseModel):
    """Normalized behavioral measurements at one point in time."""

    model_config = ConfigDict(frozen=True)

    learning: float = Field(ge=0.0, le=1.0)
    stability: float = Field(ge=0.0, le=1.0)
    retention: float = Field(ge=0.0, le=1.0)
    generalization: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    review_efficiency: float = Field(ge=0.0, le=1.0)
    contradiction_health: float = Field(ge=0.0, le=1.0)
    debate_consistency: float | None = Field(default=None, ge=0.0, le=1.0)


class IndependentEvaluator:
    """Measures forgetting, transfer, stability, and calibration without teaching."""

    def evaluate(self, inputs: EvaluationInput) -> EvaluationSnapshot:
        """Produce normalized objective scores from independent observations."""
        learning = self._learning_score(inputs)
        stability = _clamp(inputs.kel_ks if inputs.kel_ks is not None else 0.0)
        retention = self._retention_score(inputs)
        generalization = self._generalization_score(inputs)
        evidence = self._evidence_quality(inputs)
        efficiency = _clamp(1.0 - inputs.review_fraction)
        contradictions = self._contradiction_health(inputs)
        debate = None
        if inputs.debate_disagreement is not None:
            debate = _clamp(1.0 - inputs.debate_disagreement)
        return EvaluationSnapshot(
            learning=learning,
            stability=stability,
            retention=retention,
            generalization=generalization,
            evidence_quality=evidence,
            review_efficiency=efficiency,
            contradiction_health=contradictions,
            debate_consistency=debate,
        )

    @staticmethod
    def _learning_score(inputs: EvaluationInput) -> float:
        if inputs.curriculum_score is not None:
            return _clamp(inputs.curriculum_score)
        if inputs.kel_lg is not None:
            return _clamp(0.5 + inputs.kel_lg * 0.5)
        if inputs.heldout_first is not None and inputs.heldout_last is not None:
            gain = inputs.heldout_last - inputs.heldout_first
            return _clamp(0.5 + gain * 0.5)
        return 0.5

    @staticmethod
    def _retention_score(inputs: EvaluationInput) -> float:
        stability = inputs.kel_ks if inputs.kel_ks is not None else None
        forgetting = 1.0 - inputs.mean_forgetting_risk
        if stability is not None:
            return _clamp(stability * 0.6 + forgetting * 0.4)
        return _clamp(forgetting)

    @staticmethod
    def _generalization_score(inputs: EvaluationInput) -> float:
        if inputs.heldout_peak is not None and inputs.heldout_last is not None:
            if inputs.heldout_peak <= 0:
                return _clamp(inputs.heldout_last)
            return _clamp(inputs.heldout_last / inputs.heldout_peak)
        if inputs.heldout_last is not None:
            return _clamp(inputs.heldout_last)
        return 0.5

    @staticmethod
    def _evidence_quality(inputs: EvaluationInput) -> float:
        if inputs.evidence_score is not None:
            parts: list[float] = [_clamp(inputs.evidence_score)]
        else:
            parts = []
        if inputs.kel_cre is not None:
            parts.append(_clamp(inputs.kel_cre))
        if inputs.conflict_discovery is not None:
            parts.append(_clamp(1.0 - inputs.conflict_discovery))
        if inputs.missing_knowledge is not None:
            parts.append(_clamp(1.0 - inputs.missing_knowledge))
        if not parts:
            return 0.5
        return round(sum(parts) / len(parts), 4)

    @staticmethod
    def _contradiction_health(inputs: EvaluationInput) -> float:
        if inputs.kel_cd is not None:
            return _clamp(1.0 - inputs.kel_cd)
        if inputs.conflict_discovery is not None:
            return _clamp(1.0 - inputs.conflict_discovery)
        return 0.5
