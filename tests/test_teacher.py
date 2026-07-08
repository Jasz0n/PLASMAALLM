"""Tests for allm.teacher: evaluation, state, goals, progress."""

from pathlib import Path

import pytest

from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig

MATH = {"2+2?": "4", "3*3?": "9"}
GEO = {"Capital of France?": "Paris", "Capital of Japan?": "Tokyo"}


def make_samples() -> list[Sample]:
    labelled = [(*item, "math") for item in MATH.items()] + [
        (*item, "geography") for item in GEO.items()
    ]
    return [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, a, topic) in enumerate(labelled)
    ]


@pytest.fixture()
def teacher(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "state.sqlite3")
    t = Teacher(
        state=KnowledgeState(store),
        exam_generator=DatasetExamGenerator(make_samples()),
        grader=ExactMatchGrader(),
        config=TeacherConfig(confidence_smoothing=0.5, weakness_threshold=0.6),
    )
    yield t
    store.close()


def math_student() -> ScriptedStudent:
    return ScriptedStudent("math-kid", "math", knowledge=MATH)


def test_evaluate_grades_and_records(teacher: Teacher) -> None:
    exam = teacher.create_exam(num_questions=4, seed=1)
    result = teacher.evaluate(math_student(), exam)
    assert result.topic_scores()["math"] == 1.0
    assert result.topic_scores()["geography"] == 0.0
    stored = teacher.state.exam_results("math-kid")
    assert len(stored) == 1
    assert stored[0].score == result.score


def test_confidence_uses_ema(teacher: Teacher) -> None:
    student = math_student()
    exam = teacher.create_exam(topics=["geography"], num_questions=2, seed=1)
    teacher.evaluate(student, exam)
    assert teacher.state.confidence("math-kid", "geography") == 0.0

    student.learn("Capital of France?", "Paris")
    student.learn("Capital of Japan?", "Tokyo")
    teacher.evaluate(student, teacher.create_exam(topics=["geography"], num_questions=2, seed=2))
    # EMA with smoothing 0.5: 0.5 * 1.0 + 0.5 * 0.0
    assert teacher.state.confidence("math-kid", "geography") == pytest.approx(0.5)


def test_confidence_history_is_versioned(teacher: Teacher) -> None:
    student = math_student()
    for seed in (1, 2, 3):
        exam = teacher.create_exam(topics=["math"], num_questions=2, seed=seed)
        teacher.evaluate(student, exam)
    history = teacher.state.confidence_history("math-kid", "math")
    assert len(history) == 3
    assert all(confidence == 1.0 for _, confidence in history)


def test_assign_goals_targets_weakest(teacher: Teacher) -> None:
    exam = teacher.create_exam(num_questions=4, seed=1)
    teacher.evaluate(math_student(), exam)
    goals = teacher.assign_goals("math-kid")
    assert [g.topic for g in goals] == ["geography"]
    assert goals[0].priority == 1.0
    assert teacher.state.current_goals("math-kid")[0].topic == "geography"


def test_no_goals_when_all_strong(teacher: Teacher) -> None:
    exam = teacher.create_exam(topics=["math"], num_questions=2, seed=1)
    teacher.evaluate(math_student(), exam)
    assert teacher.assign_goals("math-kid") == []


def test_progress_reports_improvement(teacher: Teacher) -> None:
    student = ScriptedStudent("learner", "geography")
    exam1 = teacher.create_exam(topics=["geography"], num_questions=2, seed=1)
    teacher.evaluate(student, exam1)

    student.learn("Capital of France?", "Paris")
    student.learn("Capital of Japan?", "Tokyo")
    exam2 = teacher.create_exam(topics=["geography"], num_questions=2, seed=2)
    teacher.evaluate(student, exam2)

    report = teacher.progress("learner")
    assert report.exams_taken == 2
    geo = next(t for t in report.topics if t.topic == "geography")
    assert geo.delta > 0
    assert [t.topic for t in report.improving()] == ["geography"]


def test_global_confidence_averages_students(teacher: Teacher) -> None:
    exam = teacher.create_exam(topics=["math"], num_questions=2, seed=1)
    teacher.evaluate(math_student(), exam)
    teacher.evaluate(ScriptedStudent("novice", "geography"), exam)
    assert teacher.state.global_confidence("math") == pytest.approx(0.5)
    assert teacher.state.students() == ["math-kid", "novice"]


def test_progress_empty_student(teacher: Teacher) -> None:
    report = teacher.progress("ghost")
    assert report.exams_taken == 0
    assert report.mean_score == 0.0
    assert report.topics == ()
