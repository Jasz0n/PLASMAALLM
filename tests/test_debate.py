"""Tests for allm.debate: clustering, disagreement, verdicts, tasks."""

import pytest

from allm.debate import DebateEngine
from allm.exam import ExactMatchGrader, Question
from allm.students import ScriptedStudent


def question(expected: str | None = "Paris") -> Question:
    return Question(
        id="q1", prompt="Capital of France?", expected=expected, topic="geography"
    )


def student(sid: str, answer: str | None, confidence: float = 0.9) -> ScriptedStudent:
    knowledge = {"Capital of France?": answer} if answer is not None else {}
    return ScriptedStudent(sid, "geography", knowledge=knowledge, confident=confidence)


def test_unanimous_agreement() -> None:
    result = DebateEngine().debate(
        question(), [student("a", "Paris"), student("b", "Paris")]
    )
    assert result.disagreement == 0.0
    assert result.verdict == "Paris"
    assert not result.unresolved
    assert len(result.clusters) == 1
    assert result.clusters[0].members == ("a", "b")


def test_clustering_is_normalised() -> None:
    result = DebateEngine().debate(
        question(), [student("a", "Paris"), student("b", " paris. ")]
    )
    assert result.disagreement == 0.0


def test_confidence_weighted_verdict_beats_headcount() -> None:
    result = DebateEngine(disagreement_threshold=1.0).debate(
        question(),
        [
            student("sure", "Paris", confidence=0.95),
            student("guess1", "Lyon", confidence=0.3),
            student("guess2", "Lyon", confidence=0.3),
        ],
    )
    assert result.verdict == "Paris"  # 0.95 > 0.3 + 0.3
    assert result.disagreement == pytest.approx(1 / 3)


def test_grading_marks_positions_when_truth_known() -> None:
    engine = DebateEngine(grader=ExactMatchGrader())
    result = engine.debate(question(), [student("a", "Paris"), student("b", "Lyon")])
    by_student = {p.student_id: p.correct for p in result.positions}
    assert by_student == {"a": True, "b": False}


def test_positions_ungraded_without_truth() -> None:
    engine = DebateEngine(grader=ExactMatchGrader())
    result = engine.debate(question(expected=None), [student("a", "Paris"), student("b", "Lyon")])
    assert all(p.correct is None for p in result.positions)


def test_large_disagreement_becomes_learning_task() -> None:
    engine = DebateEngine(disagreement_threshold=0.5)
    result = engine.debate(question(), [student("a", "Paris"), student("b", "Lyon")])
    assert result.unresolved
    sample = result.to_learning_sample()
    assert sample.input == "Capital of France?"
    assert sample.target == "Paris"
    assert sample.metadata["origin"] == "debate"
    assert sample.metadata["disagreement"] == 0.5


def test_open_question_task_has_no_target() -> None:
    engine = DebateEngine(disagreement_threshold=0.4)
    result = engine.debate(question(expected=None), [student("a", "Paris"), student("b", "Lyon")])
    assert result.to_learning_sample().target is None


def test_debate_requires_two_students() -> None:
    with pytest.raises(ValueError, match="two students"):
        DebateEngine().debate(question(), [student("a", "Paris")])
