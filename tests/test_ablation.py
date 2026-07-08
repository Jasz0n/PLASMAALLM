"""Tests for controlled ablation comparison."""

from pathlib import Path

import pytest

from allm.evaluation.ablation import (
    compare_ablation_runs,
    format_ablation_report,
    load_arm_from_history,
    summarize_arm,
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


def test_compare_ablation_runs(tmp_path: Path) -> None:
    control_path = tmp_path / "control.jsonl"
    treatment_path = tmp_path / "treatment.jsonl"
    IterationHistoryWriter(control_path).append(_record(1, 0.1, 0.2))
    IterationHistoryWriter(treatment_path).append(_record(1, 0.1, 0.35))

    control = load_arm_from_history("no-mission", control_path, mission_enabled=False)
    treatment = load_arm_from_history("mission", treatment_path, mission_enabled=True)
    comparison = compare_ablation_runs(
        control,
        treatment,
        variable="student_identity",
        loop_seed=42,
    )
    assert comparison.heldout_gain_delta == pytest.approx(0.15)
    assert "student_identity" in format_ablation_report(comparison)


def test_summarize_arm_peak_score(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    writer = IterationHistoryWriter(path)
    writer.append(_record(1, 0.1, 0.2))
    writer.append(_record(2, 0.2, 0.5))
    summary = summarize_arm("mission", mission_enabled=True, records=writer.load_all(), history_path=path)
    assert summary.peak_score == 0.5
    assert summary.heldout_gain == 0.4
