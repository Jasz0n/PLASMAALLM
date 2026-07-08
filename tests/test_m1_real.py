"""Tests for EscalatingGrader, log-prob confidence, and parse robustness."""

import pytest

from allm.exam import (
    Answer,
    EscalatingGrader,
    ExactMatchGrader,
    LLMJudgeGrader,
    Question,
    build_judge_prompt,
    parse_questions,
)
from allm.exam.escalation import might_be_paraphrase
from allm.models import EchoModel, ModelSpec
from allm.students.logprob_confidence import estimate_from_logprobs

OLLAMA_STYLE = """T: Math
Q: What is the square root of 27? A: The square root of 27 is approximately 5.196.
T: Geography
Q: Which continent is located closest to China?
A: China is located in Asia, and the continent closest to it is Europe."""


def test_parse_questions_handles_inline_qa() -> None:
    triples = parse_questions(OLLAMA_STYLE)
    assert len(triples) == 2
    topics = {t.lower() for t, _, _ in triples}
    assert topics == {"math", "geography"}
    math = next(q for t, q, a in triples if t.lower() == "math")
    assert "square root" in math.lower()


def test_might_be_paraphrase_word_form_number() -> None:
    assert might_be_paraphrase("The answer is four.", "4")
    assert not might_be_paraphrase("nine", "4")
    assert not might_be_paraphrase("London", "Paris")


def test_escalating_skips_judge_for_clearly_wrong() -> None:
    calls: list[str] = []

    class SpyJudge:
        def grade(self, question, answer):
            calls.append("judge")
            raise AssertionError("should not run")

    q = Question(id="q1", prompt="Capital of France?", expected="Paris", topic="geo")
    a = Answer(question_id="q1", text="London", confidence=0.5)
    grader = EscalatingGrader(
        ExactMatchGrader("contains"),
        SpyJudge(),
        escalate_paraphrases_only=True,
    )
    result = grader.grade(q, a)
    assert not result.correct and not calls


def test_estimate_from_logprobs() -> None:
    assert estimate_from_logprobs([-0.1, -0.2]) == pytest.approx(0.861, rel=0.01)
    assert estimate_from_logprobs([]) is None


def test_escalating_skips_judge_on_exact_match() -> None:
    calls: list[str] = []

    class SpyJudge:
        def grade(self, question, answer):
            calls.append("judge")
            raise AssertionError("should not run")

    q = Question(id="q1", prompt="2+2?", expected="4", topic="math")
    a = Answer(question_id="q1", text="4", confidence=0.9)
    grader = EscalatingGrader(ExactMatchGrader("contains"), SpyJudge())
    result = grader.grade(q, a)
    assert result.correct and not calls


def test_escalating_overrides_when_judge_disagrees() -> None:
    q = Question(id="q1", prompt="Capital of France?", expected="Paris", topic="geo")
    a = Answer(question_id="q1", text="It is Paris.", confidence=0.5)
    judge_model = EchoModel(ModelSpec(name="j", provider="echo", model_id="none"))
    prompt = build_judge_prompt(q, a)
    judge_model.script(prompt, "VERDICT: correct\nSCORE: 1.0\nREASON: paraphrase ok")
    grader = EscalatingGrader(
        ExactMatchGrader("exact"),
        LLMJudgeGrader(judge_model, compare_exact=False),
    )
    result = grader.grade(q, a)
    assert result.correct
    assert result.feedback is not None and "escalated" in result.feedback
