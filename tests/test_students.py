"""Tests for allm.students (protocol + scripted double)."""

from allm.exam.base import Question
from allm.students import ScriptedStudent, Student, student_types


def question(prompt: str = "2+2?") -> Question:
    return Question(id="q1", prompt=prompt, expected="4", topic="math")


def test_satisfies_protocol() -> None:
    student = ScriptedStudent("s1", "math")
    assert isinstance(student, Student)
    assert student.student_id == "s1"
    assert student.specialty == "math"


def test_known_answer_is_confident() -> None:
    student = ScriptedStudent("s1", "math", knowledge={"2+2?": "4"})
    result = student.solve(question())
    assert result.text == "4"
    assert result.confidence == 0.9
    assert result.question_id == "q1"


def test_prompt_matching_is_normalised() -> None:
    student = ScriptedStudent("s1", "math", knowledge={"  2+2? ": "4"})
    assert student.solve(question("2+2?")).text == "4"


def test_unknown_answer_has_low_confidence() -> None:
    student = ScriptedStudent("s1", "math")
    result = student.solve(question("What is a monad?"))
    assert result.text == "I don't know"
    assert result.confidence == 0.1


def test_learn_adds_knowledge() -> None:
    student = ScriptedStudent("s1", "math")
    student.learn("2+2?", "4")
    assert student.solve(question()).text == "4"


def test_registered_type() -> None:
    assert student_types.get("scripted") is ScriptedStudent
