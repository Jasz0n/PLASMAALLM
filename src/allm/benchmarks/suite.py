"""Benchmark suite — judge ecosystem health across multiple objectives (M43)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from allm.evaluator.independent import EvaluationSnapshot
from allm.evaluator.types import EvaluationInput


class BenchmarkDimension(str, Enum):
    RETENTION = "retention"
    GENERALIZATION = "generalization"
    ADAPTATION = "adaptation"
    SCIENTIFIC_REASONING = "scientific_reasoning"
    ENGINEERING = "engineering"


class BenchmarkResult(BaseModel):
    """Score for one benchmark dimension."""

    model_config = ConfigDict(frozen=True)

    dimension: BenchmarkDimension
    score: float = Field(ge=0.0, le=1.0)
    detail: str


class BenchmarkSuite:
    """Evaluate a run against multiple goals instead of a single held-out score."""

    def evaluate(
        self,
        inputs: EvaluationInput,
        snapshot: EvaluationSnapshot | None = None,
    ) -> tuple[BenchmarkResult, ...]:
        """Score each benchmark dimension from independent evaluation."""
        from allm.evaluator.independent import IndependentEvaluator

        snap = snapshot or IndependentEvaluator().evaluate(inputs)
        return (
            BenchmarkResult(
                dimension=BenchmarkDimension.RETENTION,
                score=snap.retention,
                detail=f"KS={inputs.kel_ks} forgetting_risk={inputs.mean_forgetting_risk:.2f}",
            ),
            BenchmarkResult(
                dimension=BenchmarkDimension.GENERALIZATION,
                score=snap.generalization,
                detail=(
                    f"held-out {inputs.heldout_last} / peak {inputs.heldout_peak}"
                    if inputs.heldout_last is not None
                    else "no held-out data"
                ),
            ),
            BenchmarkResult(
                dimension=BenchmarkDimension.ADAPTATION,
                score=snap.learning,
                detail=f"LG={inputs.kel_lg} learning={snap.learning:.2f}",
            ),
            BenchmarkResult(
                dimension=BenchmarkDimension.SCIENTIFIC_REASONING,
                score=snap.evidence_quality
                if inputs.alignment_score is None
                else round((snap.evidence_quality + inputs.alignment_score) / 2, 4),
                detail=(
                    f"alignment={inputs.alignment_score} CRE={inputs.kel_cre} "
                    f"conflicts={inputs.conflict_discovery}"
                ),
            ),
            BenchmarkResult(
                dimension=BenchmarkDimension.ENGINEERING,
                score=snap.contradiction_health,
                detail=f"CD={inputs.kel_cd} debate={snap.debate_consistency}",
            ),
        )

    @staticmethod
    def net_improvement(
        before: tuple[BenchmarkResult, ...],
        after: tuple[BenchmarkResult, ...],
    ) -> dict[str, float]:
        """Per-dimension delta; positive means improvement."""
        by_dim = {row.dimension.value: row.score for row in before}
        deltas: dict[str, float] = {}
        for row in after:
            prior = by_dim.get(row.dimension.value, row.score)
            deltas[row.dimension.value] = round(row.score - prior, 4)
        return deltas

    @staticmethod
    def summarize(results: tuple[BenchmarkResult, ...]) -> str:
        lines = ["Benchmark suite:"]
        for row in results:
            lines.append(f"  {row.dimension.value:22s} {row.score:.2f}  ({row.detail})")
        return "\n".join(lines)
