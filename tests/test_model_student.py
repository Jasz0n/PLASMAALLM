"""Tests for allm.students Phase 3: confidence, ModelStudent, FailureLog."""

from pathlib import Path

import pytest

from allm.exam.base import Answer, Question
from allm.exam.grading import ExactMatchGrader
from allm.models import EchoModel, ModelSpec
from allm.students import (
    FailureLog,
    ModelStudent,
    ModelStudentConfig,
    Student,
    parse_confidence,
)
from allm.storage import SQLiteRecordStore


def echo_model(responses: dict[str, str] | None = None) -> EchoModel:
    spec = ModelSpec(name="m", provider="echo", model_id="none")
    return EchoModel(spec, responses=responses)


def question(prompt: str = "Capital of France?", qid: str = "q1") -> Question:
    return Question(id=qid, prompt=prompt, expected="Paris", topic="geography")


# -- confidence parsing ------------------------------------------------


def test_parse_confidence_extracts_and_strips() -> None:
    text, confidence = parse_confidence("Paris\nCONFIDENCE: 0.85")
    assert text == "Paris"
    assert confidence == 0.85


def test_parse_confidence_clamps() -> None:
    assert parse_confidence("x\nCONFIDENCE: 7")[1] == 1.0


def test_parse_confidence_missing_returns_none() -> None:
    text, confidence = parse_confidence("just an answer")
    assert text == "just an answer"
    assert confidence is None


def test_parse_confidence_case_insensitive() -> None:
    assert parse_confidence("y\nconfidence: 0.4")[1] == 0.4


# -- ModelStudent ------------------------------------------------------


def test_satisfies_student_protocol() -> None:
    student = ModelStudent("s1", "geography", echo_model())
    assert isinstance(student, Student)


def test_solve_parses_model_confidence() -> None:
    student = ModelStudent("s1", "geography", echo_model())
    prompt = student.build_prompt(question())
    student_model = echo_model({prompt: "Paris\nCONFIDENCE: 0.8"})
    student = ModelStudent("s1", "geography", student_model)
    answer = student.solve(question())
    assert answer.text == "Paris"
    assert answer.confidence == 0.8


def test_solve_applies_default_confidence_when_unparseable() -> None:
    student = ModelStudent(
        "s1", "geography", echo_model(), ModelStudentConfig(default_confidence=0.25)
    )
    answer = student.solve(question())
    assert answer.confidence == 0.25  # echo fallback has no CONFIDENCE line


def test_studied_note_answers_from_memory() -> None:
    student = ModelStudent("s1", "geography", echo_model())
    student.study("Capital of France?", "Paris")
    answer = student.solve(question())
    assert answer.text == "Paris"
    assert answer.confidence == ModelStudentConfig().memory_confidence
    assert answer.reasoning == "retrieved from studied notes"


def test_notes_appear_in_prompt() -> None:
    student = ModelStudent("s1", "geography", echo_model())
    student.study("Capital of Japan?", "Tokyo")
    prompt = student.build_prompt(question())
    assert "Q: Capital of Japan? A: Tokyo" in prompt


def test_notes_are_bounded_fifo() -> None:
    student = ModelStudent(
        "s1", "geography", echo_model(), ModelStudentConfig(max_notes=2)
    )
    for i in range(3):
        student.study(f"question {i}?", f"answer {i}")
    remembered = [q for q, _ in student.notes]
    assert remembered == ["question 1?", "question 2?"]


def test_restudying_updates_answer() -> None:
    student = ModelStudent("s1", "geography", echo_model())
    student.study("Capital of France?", "Lyon")
    student.study("Capital of France?", "Paris")
    assert student.solve(question()).text == "Paris"


# -- FailureLog --------------------------------------------------------


@pytest.fixture()
def failure_log(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "failures.sqlite3")
    yield FailureLog(store)
    store.close()


def graded_failure(qid: str = "q1"):
    grader = ExactMatchGrader()
    return grader.grade(
        question(qid=qid), Answer(question_id=qid, text="London", confidence=0.7)
    )


def test_record_and_list_failures(failure_log: FailureLog) -> None:
    failure_log.record("s1", graded_failure())
    failures = failure_log.failures("s1")
    assert len(failures) == 1
    assert failures[0].given == "London"
    assert failures[0].expected == "Paris"
    assert failures[0].confidence == 0.7
    assert failure_log.failures("someone-else") == []


def test_failures_become_training_samples(failure_log: FailureLog) -> None:
    failure_log.record("s1", graded_failure())
    samples = failure_log.training_samples("s1")
    assert len(samples) == 1
    assert samples[0].input == "Capital of France?"
    assert samples[0].target == "Paris"
    assert samples[0].metadata["origin"] == "failure"


def test_repeat_failure_is_versioned_not_duplicated(failure_log: FailureLog) -> None:
    failure_log.record("s1", graded_failure())
    failure_log.record("s1", graded_failure())
    assert len(failure_log.failures("s1")) == 1  # latest version per question
