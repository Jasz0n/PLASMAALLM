"""Compare controlled experiment runs (one variable isolated)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from allm.evaluation.strategy_gain import StrategyPhaseGain, compute_marginal_strategy_gains
from allm.loop.history import IterationHistoryWriter, LearningIterationRecord


class AblationArmSummary(BaseModel):
    """Summary metrics for one experiment arm."""

    model_config = ConfigDict(frozen=True)

    label: str
    mission_enabled: bool
    first_score: float
    last_score: float
    heldout_gain: float
    iterations_completed: int
    peak_score: float
    kel_lg: float | None = None
    history_path: str
    phase_gains: tuple[StrategyPhaseGain, ...] = ()


class AblationComparison(BaseModel):
    """Side-by-side comparison of two arms that differ in one knob."""

    model_config = ConfigDict(frozen=True)

    variable: str
    loop_seed: int
    control: AblationArmSummary
    treatment: AblationArmSummary
    heldout_gain_delta: float
    peak_score_delta: float


def summarize_arm(
    label: str,
    *,
    mission_enabled: bool,
    records: list[LearningIterationRecord],
    history_path: Path | str,
    kel_lg: float | None = None,
) -> AblationArmSummary:
    """Build a summary from iteration history records."""
    if not records:
        return AblationArmSummary(
            label=label,
            mission_enabled=mission_enabled,
            first_score=0.0,
            last_score=0.0,
            heldout_gain=0.0,
            iterations_completed=0,
            peak_score=0.0,
            kel_lg=kel_lg,
            history_path=str(history_path),
        )

    ordered = sorted(records, key=lambda row: row.iteration)
    scores = [row.score_after for row in ordered]
    first = ordered[0].score_before
    last = ordered[-1].score_after
    return AblationArmSummary(
        label=label,
        mission_enabled=mission_enabled,
        first_score=first,
        last_score=last,
        heldout_gain=last - first,
        iterations_completed=len(ordered),
        peak_score=max(scores) if scores else 0.0,
        kel_lg=kel_lg,
        history_path=str(history_path),
        phase_gains=tuple(compute_marginal_strategy_gains(ordered)),
    )


def compare_ablation_runs(
    control: AblationArmSummary,
    treatment: AblationArmSummary,
    *,
    variable: str,
    loop_seed: int,
) -> AblationComparison:
    """Compare treatment against control on held-out transfer."""
    return AblationComparison(
        variable=variable,
        loop_seed=loop_seed,
        control=control,
        treatment=treatment,
        heldout_gain_delta=treatment.heldout_gain - control.heldout_gain,
        peak_score_delta=treatment.peak_score - control.peak_score,
    )


def load_arm_from_history(
    label: str,
    history_path: Path | str,
    *,
    mission_enabled: bool,
) -> AblationArmSummary:
    """Load one arm summary from a ``learning_history.jsonl`` file."""
    records = IterationHistoryWriter(history_path).load_all()
    kel_lg = records[-1].kel_lg if records else None
    return summarize_arm(
        label,
        mission_enabled=mission_enabled,
        records=records,
        history_path=history_path,
        kel_lg=kel_lg,
    )


def export_ablation_comparison(path: Path | str, comparison: AblationComparison) -> Path:
    """Write comparison JSON for later analysis."""
    target = Path(path)
    target.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    return target


def format_ablation_report(comparison: AblationComparison) -> str:
    """Human-readable ablation table."""
    control = comparison.control
    treatment = comparison.treatment
    lines = [
        f"Variable: {comparison.variable}  (loop seed={comparison.loop_seed})",
        "",
        f"  {'Arm':<14} {'Mission':>8} {'First':>7} {'Last':>7} {'Gain':>7} {'Peak':>7} {'Iters':>6}",
        f"  {'-' * 58}",
        f"  {control.label:<14} {str(control.mission_enabled):>8} "
        f"{control.first_score:>7.2f} {control.last_score:>7.2f} "
        f"{control.heldout_gain:>+7.2f} {control.peak_score:>7.2f} {control.iterations_completed:>6}",
        f"  {treatment.label:<14} {str(treatment.mission_enabled):>8} "
        f"{treatment.first_score:>7.2f} {treatment.last_score:>7.2f} "
        f"{treatment.heldout_gain:>+7.2f} {treatment.peak_score:>7.2f} {treatment.iterations_completed:>6}",
        "",
        f"  Δ held-out gain: {comparison.heldout_gain_delta:+.2f}",
        f"  Δ peak score:    {comparison.peak_score_delta:+.2f}",
    ]
    return "\n".join(lines)
