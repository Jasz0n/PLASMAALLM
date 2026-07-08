"""Tests for Phase 7: generative exams, coding grader, composite routing."""

import pytest

from allm.exam import (
    Answer,
    CodingGrader,
    CompositeGrader,
    ExactMatchGrader,
    ModelExamGenerator,
    Question,
    extract_code,
    parse_questions,
)
from allm.models import EchoModel, ModelSpec


def echo(responses: dict[str, str] | None = None) -> EchoModel:
    return EchoModel(ModelSpec(name="m", provider="echo", model_id="none"), responses)


MODEL_OUTPUT = """T: math
Q: What is 2+2?
A: 4
T: physics
Q: What force pulls the apple down?
A: gravity
"""


# -- parsing and generation ---------------------------------------------


def test_parse_questions() -> None:
    triples = parse_questions(MODEL_OUTPUT)
    assert triples == [
        ("math", "What is 2+2?", "4"),
        ("physics", "What force pulls the apple down?", "gravity"),
    ]
    assert parse_questions("no structure here") == []


def test_generator_builds_exam_from_model_output() -> None:
    generator = ModelExamGenerator(echo())
    prompt = generator.build_prompt(["math", "physics"], 2)
    generator = ModelExamGenerator(echo({prompt: MODEL_OUTPUT}))
    exam = generator.generate(topics=["math", "physics"], num_questions=2)
    assert [q.topic for q in exam.questions] == ["math", "physics"]
    assert exam.questions[0].expected == "4"
    assert exam.questions[0].kind == "factual"


def test_generator_truncates_excess_questions() -> None:
    generator = ModelExamGenerator(echo())
    prompt = generator.build_prompt(["math"], 1)
    generator = ModelExamGenerator(echo({prompt: MODEL_OUTPUT}))
    exam = generator.generate(topics=["math"], num_questions=1)
    assert len(exam.questions) == 1


def test_generator_rejects_unparseable_output() -> None:
    generator = ModelExamGenerator(echo())
    prompt = generator.build_prompt(["math"], 2)
    generator = ModelExamGenerator(echo({prompt: "I refuse to write exams."}))
    with pytest.raises(ValueError, match="parseable"):
        generator.generate(topics=["math"], num_questions=2)


def test_prompt_carries_kind_and_difficulty() -> None:
    generator = ModelExamGenerator(echo(), kind="cross_domain", difficulty=4)
    prompt = generator.build_prompt(["math", "music"], 3)
    assert "combine at least two" in prompt
    assert "Difficulty level: 4" in prompt


def test_difficulty_must_be_positive() -> None:
    with pytest.raises(ValueError, match="difficulty"):
        ModelExamGenerator(echo(), difficulty=0)


# -- coding grader -------------------------------------------------------


def coding_question(expected: str = "42") -> Question:
    return Question(
        id="c1", prompt="Print the answer.", expected=expected, topic="python", kind="coding"
    )


def code_answer(code: str) -> Answer:
    return Answer(question_id="c1", text=code, confidence=0.5)


def test_correct_program_passes() -> None:
    result = CodingGrader().grade(coding_question(), code_answer("print(6*7)"))
    assert result.correct


def test_fenced_code_is_extracted() -> None:
    assert extract_code("```python\nprint(1)\n```") == "print(1)"
    assert extract_code("print(2)") == "print(2)"
    result = CodingGrader().grade(
        coding_question(), code_answer("```python\nprint(42)\n```")
    )
    assert result.correct


def test_wrong_output_fails_with_feedback() -> None:
    result = CodingGrader().grade(coding_question(), code_answer("print(41)"))
    assert not result.correct
    assert "41" in result.feedback


def test_crash_reports_error() -> None:
    result = CodingGrader().grade(coding_question(), code_answer("1/0"))
    assert not result.correct
    assert "ZeroDivisionError" in result.feedback


def test_infinite_loop_times_out() -> None:
    grader = CodingGrader(timeout_seconds=1.0)
    result = grader.grade(coding_question(), code_answer("while True: pass"))
    assert not result.correct
    assert "timed out" in result.feedback


# -- composite routing ----------------------------------------------------


def test_composite_routes_by_kind() -> None:
    grader = CompositeGrader(ExactMatchGrader(), {"coding": CodingGrader()})
    factual = Question(id="f1", prompt="2+2?", expected="4", topic="math")
    assert grader.grade(factual, Answer(question_id="f1", text="4", confidence=1)).correct
    assert grader.grade(coding_question(), code_answer("print(42)")).correct
    assert not grader.grade(coding_question(), code_answer("42")).correct  # ran, printed nothing
