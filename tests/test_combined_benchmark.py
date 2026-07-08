"""Tests for combined-source KEL benchmark reporting."""

from pathlib import Path

import pytest

from allm.evaluation.combined_benchmark import (
    compare_combined_benchmark_runs,
    format_combined_benchmark_report,
    metrics_from_records,
)
from allm.loop.history import IterationHistoryWriter, LearningIterationRecord


def _record(iteration: int, before: float, after: float) -> LearningIterationRecord:
    return LearningIterationRecord(
        iteration=iteration,
        strategy="definitions",
        sample_kinds=("definition",),
        sample_ids=("s1",),
        student_id="kids-kel",
        score_before=before,
        score_after=after,
        goals=("kids-plasma",),
        samples_studied=32,
        questions_per_exam=8,
        use_exam_paraphrase=False,
        holdout_answers_in_train=0,
    )


def test_compare_combined_benchmark_runs(tmp_path: Path) -> None:
    control_path = tmp_path / "control.jsonl"
    treatment_path = tmp_path / "treatment.jsonl"
    IterationHistoryWriter(control_path).append(_record(1, 0.1, 0.2))
    IterationHistoryWriter(treatment_path).append(_record(1, 0.1, 0.35))

    control = metrics_from_records(
        "no-visual",
        visual_delivery_enabled=False,
        records=IterationHistoryWriter(control_path).load_all(),
        history_path=control_path,
        kel_lg=0.02,
        visual_notes_delivered=0,
    )
    treatment = metrics_from_records(
        "visual",
        visual_delivery_enabled=True,
        records=IterationHistoryWriter(treatment_path).load_all(),
        history_path=treatment_path,
        kel_lg=0.08,
        aligned_concepts=4,
        visual_notes_delivered=12,
        teacher_approved_briefs=2,
        export_mode="teacher_ui",
    )
    comparison = compare_combined_benchmark_runs(
        control,
        treatment,
        variable="visual_delivery",
        loop_seed=42,
    )
    assert comparison.heldout_gain_delta == pytest.approx(0.15)
    assert comparison.kel_lg_delta == pytest.approx(0.06)
    assert comparison.visual_notes_delta == 12
    assert "visual_delivery" in format_combined_benchmark_report(comparison)
