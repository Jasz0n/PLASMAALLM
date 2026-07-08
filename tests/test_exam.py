"""Tests for allm.exam: vocabulary, dataset generation, grading."""

import pytest

from allm.data.base import Sample
from allm.exam import (
    Answer,
    DatasetExamGenerator,
    ExactMatchGrader,
    ExamResult,
    Question,
    exam_generators,
    graders,
    normalise,
)


def make_samples() -> list[Sample]:
    rows = [
        ("2+2?", "4", "math"),
        ("3*3?", "9", "math"),
        ("10/2?", "5", "math"),
        ("Capital of France?", "Paris", "geography"),
        ("Capital of Japan?", "Tokyo", "geography"),
    ]
    return [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, a, topic) in enumerate(rows)
    ]


# -- generator ---------------------------------------------------------


def test_generator_registered() -> None:
    assert exam_generators.get("dataset") is DatasetExamGenerator


def test_generate_filters_by_topic() -> None:
    gen = DatasetExamGenerator(make_samples())
    exam = gen.generate(topics=["math"], num_questions=10)
    assert len(exam.questions) == 3
    assert exam.topics() == ["math"]


def test_generate_is_seed_reproducible() -> None:
    gen = DatasetExamGenerator(make_samples())
    first = gen.generate(num_questions=3, seed=42)
    second = gen.generate(num_questions=3, seed=42)
    assert [q.prompt for q in first.questions] == [q.prompt for q in second.questions]
    assert first.id != second.id  # ids stay unique


def test_generator_rejects_unlabelled_samples() -> None:
    samples = [Sample(id="s0", input="open question", target=None)]
    with pytest.raises(ValueError, match="without targets"):
        DatasetExamGenerator(samples)


def test_generator_rejects_unknown_topic() -> None:
    gen = DatasetExamGenerator(make_samples())
    with pytest.raises(ValueError, match="biology"):
        gen.generate(topics=["biology"])


def test_topics_listing() -> None:
    assert DatasetExamGenerator(make_samples()).topics() == ["geography", "math"]


# -- grading -----------------------------------------------------------


def question(expected: str | None = "Paris") -> Question:
    return Question(id="q1", prompt="Capital of France?", expected=expected, topic="geo")


def answer(text: str) -> Answer:
    return Answer(question_id="q1", text=text, confidence=0.5)


def test_normalise() -> None:
    assert normalise("  The QUICK   fox. ") == "the quick fox"


def test_contains_mode_accepts_chatty_answers() -> None:
    result = ExactMatchGrader("contains").grade(question(), answer("It is Paris, of course."))
    assert result.correct and result.score == 1.0


def test_exact_mode_rejects_chatty_answers() -> None:
    result = ExactMatchGrader("exact").grade(question(), answer("It is Paris."))
    assert not result.correct
    assert "expected" in result.feedback


def test_ungradeable_question_scores_zero_with_feedback() -> None:
    result = ExactMatchGrader().grade(question(expected=None), answer("anything"))
    assert not result.correct
    assert "judge" in result.feedback


def test_grader_registered() -> None:
    assert graders.get("exact_match") is ExactMatchGrader


# -- results -----------------------------------------------------------


def test_exam_result_aggregation() -> None:
    grader = ExactMatchGrader()
    q_math = Question(id="a", prompt="2+2?", expected="4", topic="math")
    q_geo = Question(id="b", prompt="Capital of France?", expected="Paris", topic="geo")
    results = (
        grader.grade(q_math, Answer(question_id="a", text="4", confidence=0.9)),
        grader.grade(q_geo, Answer(question_id="b", text="London", confidence=0.4)),
    )
    exam_result = ExamResult(exam_id="e1", student_id="s1", results=results)
    assert exam_result.score == 0.5
    assert exam_result.topic_scores() == {"math": 1.0, "geo": 0.0}
    assert [r.question.id for r in exam_result.failures()] == ["b"]


def test_empty_exam_result_scores_zero() -> None:
    assert ExamResult(exam_id="e", student_id="s", results=()).score == 0.0
