"""Tests for allm.evaluation metrics."""

from pathlib import Path

import pytest

from allm.data.base import Sample
from allm.evaluation import evaluate_student, self_correction_rate
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.memory import EpisodicMemory
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig

FACTS = {"2+2?": "4", "3*3?": "9"}


@pytest.fixture()
def env(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "eval.sqlite3")
    samples = [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": "math"})
        for i, (q, a) in enumerate(FACTS.items())
    ]
    teacher = Teacher(
        KnowledgeState(store),
        DatasetExamGenerator(samples),
        ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=1.0),  # latest exam is the whole truth
    )
    yield teacher, EpisodicMemory(store)
    store.close()


def test_metrics_track_a_learning_student(env) -> None:
    teacher, memory = env
    student = ScriptedStudent("kid", "math")

    result = teacher.evaluate(student, teacher.create_exam(num_questions=2, seed=1))
    memory.remember_exam(result)  # all failures

    for question, answer in FACTS.items():
        student.learn(question, answer)
    result = teacher.evaluate(student, teacher.create_exam(num_questions=2, seed=2))
    memory.remember_exam(result)  # all successes

    evaluation = evaluate_student(teacher.state, memory, "kid")
    assert evaluation.improvement_per_topic == {"math": 1.0}
    assert evaluation.learning_speed == 1.0
    assert evaluation.mastery == 1.0
    assert evaluation.self_correction_rate == 1.0


def test_self_correction_none_without_failures(env) -> None:
    teacher, memory = env
    student = ScriptedStudent("ace", "math", knowledge=dict(FACTS))
    memory.remember_exam(
        teacher.evaluate(student, teacher.create_exam(num_questions=2, seed=1))
    )
    assert self_correction_rate(memory, "ace") is None


def test_partial_self_correction(env) -> None:
    teacher, memory = env
    student = ScriptedStudent("kid", "math")
    memory.remember_exam(
        teacher.evaluate(student, teacher.create_exam(num_questions=2, seed=1))
    )
    student.learn("2+2?", "4")  # learns only one of two failures
    memory.remember_exam(
        teacher.evaluate(student, teacher.create_exam(num_questions=2, seed=2))
    )
    assert self_correction_rate(memory, "kid") == 0.5


def test_fresh_student_has_empty_metrics(env) -> None:
    teacher, memory = env
    evaluation = evaluate_student(teacher.state, memory, "ghost")
    assert evaluation.improvement_per_topic == {}
    assert evaluation.learning_speed == 0.0
    assert evaluation.mastery == 0.0
    assert evaluation.self_correction_rate is None
