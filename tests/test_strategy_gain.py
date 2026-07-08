"""Tests for marginal learning gain per strategy phase."""

import pytest

from allm.evaluation.strategy_gain import (
    compute_marginal_strategy_gains,
    export_strategy_phase_gains,
    format_strategy_gain_report,
)
from allm.loop.history import LearningIterationRecord


def _record(
    iteration: int,
    strategy: str,
    *,
    score_before: float,
    score_after: float,
    samples_studied: int,
    kel_lg: float | None = None,
) -> LearningIterationRecord:
    return LearningIterationRecord(
        iteration=iteration,
        strategy=strategy,  # type: ignore[arg-type]
        sample_kinds=("definition",),
        sample_ids=("s1",),
        student_id="kid",
        score_before=score_before,
        score_after=score_after,
        goals=("kids-plasma",),
        samples_studied=samples_studied,
        questions_per_exam=8,
        use_exam_paraphrase=False,
        kel_lg=kel_lg,
    )


def test_single_strategy_phase() -> None:
    records = [
        _record(1, "definitions", score_before=0.1, score_after=0.0, samples_studied=32, kel_lg=-0.05),
        _record(2, "definitions", score_before=0.12, score_after=0.35, samples_studied=32, kel_lg=0.11),
    ]
    phases = compute_marginal_strategy_gains(records)
    assert len(phases) == 1
    assert phases[0].strategy == "definitions"
    assert phases[0].samples_studied == 64
    assert phases[0].heldout_gain == pytest.approx(0.25)
    assert phases[0].kel_lg_delta == pytest.approx(0.16)


def test_multiple_strategy_phases() -> None:
    records = [
        _record(1, "definitions", score_before=0.11, score_after=0.35, samples_studied=128, kel_lg=0.11),
        _record(2, "relations", score_before=0.12, score_after=0.0, samples_studied=32, kel_lg=-0.02),
        _record(3, "research", score_before=0.23, score_after=0.62, samples_studied=137, kel_lg=0.30),
    ]
    phases = compute_marginal_strategy_gains(records)
    assert [phase.strategy for phase in phases] == ["definitions", "relations", "research"]
    assert phases[0].heldout_gain == 0.24
    assert phases[1].heldout_gain == -0.12
    assert phases[2].heldout_gain == 0.39


def test_format_and_export(tmp_path) -> None:
    records = [
        _record(1, "definitions", score_before=0.0, score_after=0.35, samples_studied=120, kel_lg=0.1),
    ]
    text = format_strategy_gain_report(compute_marginal_strategy_gains(records))
    assert "definitions" in text
    assert "+0.35" in text

    out = export_strategy_phase_gains(tmp_path / "strategy_phase_gains.json", records)
    assert out.is_file()
    assert '"strategy": "definitions"' in out.read_text(encoding="utf-8")
