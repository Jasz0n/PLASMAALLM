"""Benchmark reporting for combined workshop + book KEL runs (M30)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from allm.evaluation.ablation import summarize_arm
from allm.evaluation.strategy_gain import StrategyPhaseGain, compute_marginal_strategy_gains
from allm.loop.history import IterationHistoryWriter, LearningIterationRecord


class CombinedSourcesRunMetrics(BaseModel):
    """Held-out scores plus Researcher multimodal metrics for one arm."""

    model_config = ConfigDict(frozen=True)

    label: str
    visual_delivery_enabled: bool
    first_score: float
    last_score: float
    heldout_gain: float
    peak_score: float
    iterations_completed: int
    kel_lg: float | None = None
    workshop_packages: int = 0
    book_packages: int = 0
    aligned_concepts: int = 0
    book_figures: int = 0
    student_visual_exports: int = 0
    visual_notes_delivered: int = 0
    teacher_ui_approval: bool = False
    teacher_approved_briefs: int = 0
    teacher_rejected_briefs: int = 0
    export_mode: str = "auto"
    multimodal_synced: int = 0
    history_path: str
    phase_gains: tuple[StrategyPhaseGain, ...] = Field(default_factory=tuple)


class CombinedSourcesBenchmarkComparison(BaseModel):
    """Side-by-side comparison of two combined-source arms."""

    model_config = ConfigDict(frozen=True)

    variable: str
    loop_seed: int
    control: CombinedSourcesRunMetrics
    treatment: CombinedSourcesRunMetrics
    heldout_gain_delta: float
    peak_score_delta: float
    kel_lg_delta: float | None = None
    visual_notes_delta: int = 0


def metrics_from_records(
    label: str,
    *,
    visual_delivery_enabled: bool,
    records: list[LearningIterationRecord],
    history_path: Path | str,
    kel_lg: float | None = None,
    workshop_packages: int = 0,
    book_packages: int = 0,
    aligned_concepts: int = 0,
    book_figures: int = 0,
    student_visual_exports: int = 0,
    visual_notes_delivered: int = 0,
    teacher_ui_approval: bool = False,
    teacher_approved_briefs: int = 0,
    teacher_rejected_briefs: int = 0,
    export_mode: str = "auto",
    multimodal_synced: int = 0,
) -> CombinedSourcesRunMetrics:
    """Build combined-source metrics from iteration history and Researcher counts."""
    arm = summarize_arm(
        label,
        mission_enabled=True,
        records=records,
        history_path=history_path,
        kel_lg=kel_lg,
    )
    return CombinedSourcesRunMetrics(
        label=label,
        visual_delivery_enabled=visual_delivery_enabled,
        first_score=arm.first_score,
        last_score=arm.last_score,
        heldout_gain=arm.heldout_gain,
        peak_score=arm.peak_score,
        iterations_completed=arm.iterations_completed,
        kel_lg=arm.kel_lg,
        workshop_packages=workshop_packages,
        book_packages=book_packages,
        aligned_concepts=aligned_concepts,
        book_figures=book_figures,
        student_visual_exports=student_visual_exports,
        visual_notes_delivered=visual_notes_delivered,
        teacher_ui_approval=teacher_ui_approval,
        teacher_approved_briefs=teacher_approved_briefs,
        teacher_rejected_briefs=teacher_rejected_briefs,
        export_mode=export_mode,
        multimodal_synced=multimodal_synced,
        history_path=str(history_path),
        phase_gains=tuple(compute_marginal_strategy_gains(records)),
    )


def metrics_from_kel_result(
    result: object,
    *,
    label: str,
    visual_delivery_enabled: bool,
    export_mode: str | None = None,
) -> CombinedSourcesRunMetrics:
    """Build metrics from a ``KidsKelRunResult`` (examples runner)."""
    mode = export_mode
    if mode is None:
        mode = "teacher_ui" if getattr(result, "teacher_approved_briefs", 0) else "auto"
    records = IterationHistoryWriter(result.history_path).load_all()
    return metrics_from_records(
        label,
        visual_delivery_enabled=visual_delivery_enabled,
        records=records,
        history_path=result.history_path,
        kel_lg=result.kel_lg,
        workshop_packages=result.workshop_packages,
        book_packages=result.book_packages,
        aligned_concepts=result.aligned_concepts,
        book_figures=result.book_figures,
        student_visual_exports=result.student_visual_exports,
        visual_notes_delivered=result.visual_notes_delivered,
        teacher_ui_approval=mode == "teacher_ui",
        teacher_approved_briefs=getattr(result, "teacher_approved_briefs", 0),
        teacher_rejected_briefs=getattr(result, "teacher_rejected_briefs", 0),
        export_mode=mode,
        multimodal_synced=result.multimodal_synced,
    )


def compare_combined_benchmark_runs(
    control: CombinedSourcesRunMetrics,
    treatment: CombinedSourcesRunMetrics,
    *,
    variable: str,
    loop_seed: int,
) -> CombinedSourcesBenchmarkComparison:
    """Compare treatment against control on held-out transfer and delivery."""
    kel_delta: float | None = None
    if control.kel_lg is not None and treatment.kel_lg is not None:
        kel_delta = treatment.kel_lg - control.kel_lg
    return CombinedSourcesBenchmarkComparison(
        variable=variable,
        loop_seed=loop_seed,
        control=control,
        treatment=treatment,
        heldout_gain_delta=treatment.heldout_gain - control.heldout_gain,
        peak_score_delta=treatment.peak_score - control.peak_score,
        kel_lg_delta=kel_delta,
        visual_notes_delta=treatment.visual_notes_delivered - control.visual_notes_delivered,
    )


def export_combined_benchmark(path: Path | str, comparison: CombinedSourcesBenchmarkComparison) -> Path:
    """Write benchmark comparison JSON for later analysis."""
    target = Path(path)
    target.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    return target


def format_combined_benchmark_report(comparison: CombinedSourcesBenchmarkComparison) -> str:
    """Human-readable combined-source benchmark table."""
    control = comparison.control
    treatment = comparison.treatment
    lines = [
        f"Variable: {comparison.variable}  (loop seed={comparison.loop_seed})",
        "",
        f"  {'Arm':<18} {'Export':>10} {'First':>7} {'Last':>7} {'Gain':>7} {'Peak':>7} "
        f"{'KEL LG':>7} {'VisNotes':>8} {'T.Appr':>6}",
        f"  {'-' * 88}",
        _format_arm_row(control),
        _format_arm_row(treatment),
        "",
        f"  Δ held-out gain:   {comparison.heldout_gain_delta:+.2f}",
        f"  Δ peak score:      {comparison.peak_score_delta:+.2f}",
    ]
    if comparison.kel_lg_delta is not None:
        lines.append(f"  Δ KEL learning gain: {comparison.kel_lg_delta:+.3f}")
    lines.append(f"  Δ visual notes:      {comparison.visual_notes_delta:+d}")
    lines.extend(
        [
            "",
            "  Researcher (treatment arm):",
            f"    workshop packages: {treatment.workshop_packages}",
            f"    book packages:     {treatment.book_packages}",
            f"    book figures:      {treatment.book_figures}",
            f"    visual exports:    {treatment.student_visual_exports}",
            f"    Teacher approved:  {treatment.teacher_approved_briefs}",
            f"    Teacher rejected:  {treatment.teacher_rejected_briefs}",
            f"    multimodal synced: {treatment.multimodal_synced}",
        ]
    )
    return "\n".join(lines)


def _format_arm_row(arm: CombinedSourcesRunMetrics) -> str:
    kel = f"{arm.kel_lg:.3f}" if arm.kel_lg is not None else "   n/a"
    export = arm.export_mode[:10]
    return (
        f"  {arm.label:<18} {export:>10} "
        f"{arm.first_score:>7.2f} {arm.last_score:>7.2f} "
        f"{arm.heldout_gain:>+7.2f} {arm.peak_score:>7.2f} "
        f"{kel:>7} {arm.visual_notes_delivered:>8} {arm.teacher_approved_briefs:>6}"
    )
