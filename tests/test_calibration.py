"""Tests for confidence calibration metrics."""

import pytest
from datetime import datetime, timezone

from allm.evaluation.calibration import calibration_report
from allm.exam.base import Answer, ExamResult, Question, QuestionResult
from allm.storage import SQLiteRecordStore
from allm.teacher.state import KnowledgeState


def test_calibration_report_buckets() -> None:
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    q = Question(id="q1", prompt="2+2?", expected="4", topic="math")
    results = (
        QuestionResult(
            question=q,
            answer=Answer(question_id="q1", text="4", confidence=0.9),
            score=1.0,
            correct=True,
        ),
        QuestionResult(
            question=q,
            answer=Answer(question_id="q1", text="5", confidence=0.8),
            score=0.0,
            correct=False,
        ),
    )
    exam = ExamResult(
        exam_id="e1",
        student_id="s1",
        results=results,
        taken_at=datetime.now(timezone.utc),
    )
    state.record_exam_result(exam, smoothing=1.0)
    report = calibration_report(state, "s1")
    assert report.n_samples == 2
    assert report.brier_score == pytest.approx(0.325, abs=0.01)
    assert sum(b.count for b in report.buckets) == 2
    store.close()
