"""Tests for LLM-judge grading (mocked model — no API calls)."""

from allm.exam import Answer, LLMJudgeGrader, Question, build_judge_prompt, parse_judge_response
from allm.models import EchoModel, ModelSpec


def echo_judge(responses: dict[str, str]) -> LLMJudgeGrader:
    model = EchoModel(ModelSpec(name="judge", provider="echo", model_id="none"), responses)
    return LLMJudgeGrader(model, compare_exact=True)


def test_parse_judge_response() -> None:
    text = "VERDICT: correct\nSCORE: 0.95\nREASON: matches the reference"
    correct, score, reason = parse_judge_response(text)
    assert correct and score == 0.95 and reason == "matches the reference"


def test_build_judge_prompt_includes_reference() -> None:
    q = Question(id="q1", prompt="2+2?", expected="4", topic="math")
    a = Answer(question_id="q1", text="four", confidence=0.5)
    prompt = build_judge_prompt(q, a)
    assert "2+2?" in prompt and "4" in prompt and "four" in prompt


def test_judge_grades_via_model() -> None:
    q = Question(id="q1", prompt="Capital of France?", expected="Paris", topic="geo")
    a = Answer(question_id="q1", text="It is Paris.", confidence=0.6)
    grader = echo_judge({})
    grader._model.script(
        build_judge_prompt(q, a),
        "VERDICT: correct\nSCORE: 1.0\nREASON: equivalent",
    )
    result = grader.grade(q, a)
    assert result.correct and result.score == 1.0


def test_judge_disagreement_surfaces_in_feedback() -> None:
    q = Question(id="q1", prompt="Capital of France?", expected="Paris", topic="geo")
    a = Answer(question_id="q1", text="London", confidence=0.9)
    grader = echo_judge({})
    grader._model.script(
        build_judge_prompt(q, a),
        "VERDICT: correct\nSCORE: 1.0\nREASON: generous judge",
    )
    result = grader.grade(q, a)
    assert result.correct
    assert result.feedback is not None and "disagree" in result.feedback
