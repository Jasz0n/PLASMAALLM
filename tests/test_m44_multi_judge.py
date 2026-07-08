"""Tests for M44 multi-dimensional grading."""

from __future__ import annotations

import json
from pathlib import Path

from allm.exam import (
    Answer,
    EscalatingGrader,
    ExactMatchGrader,
    ExamResult,
    MultiDimensionalGrader,
    Question,
    build_multi_judge_prompt,
    collect_disagreements,
    parse_multi_judge_response,
)
from allm.exam.disagreement_log import append_disagreements
from allm.models import EchoModel, ModelSpec


def _multi_response(
    *,
    curriculum: str = "correct",
    curriculum_score: str = "0.95",
    alignment: str = "partially_aligned",
    evidence: str = "0.42",
) -> str:
    return (
        f"CURRICULUM_VERDICT: {curriculum}\n"
        f"CURRICULUM_SCORE: {curriculum_score}\n"
        f"CURRICULUM_REASON: captures curriculum definition\n"
        f"ALIGNMENT: {alignment}\n"
        f"ALIGNMENT_SCORE: 0.35\n"
        f"ALIGNMENT_REASON: differs from mainstream terminology\n"
        f"EVIDENCE_SCORE: {evidence}\n"
        f"EVIDENCE_REASON: supported in supplied reference only\n"
    )


def test_parse_multi_judge_response() -> None:
    verdict = parse_multi_judge_response(_multi_response())
    assert verdict.curriculum_correct
    assert verdict.curriculum_score == 0.95
    assert verdict.alignment == "partially_aligned"
    assert verdict.evidence_score == 0.42


def test_curriculum_prompt_restricts_outside_knowledge() -> None:
    q = Question(id="q1", prompt="What is plasma?", expected="dynamic matter and fields", topic="plasma")
    a = Answer(question_id="q1", text="free ions and magnetic fields", confidence=0.5)
    prompt = build_multi_judge_prompt(q, a)
    assert "ONLY the reference answer" in prompt
    assert "Do NOT use outside" in prompt


def test_multi_judge_grader_attaches_verdict() -> None:
    q = Question(id="q1", prompt="What is plasma?", expected="dynamic matter and fields", topic="plasma")
    a = Answer(question_id="q1", text="free ions and fields", confidence=0.5)
    model = EchoModel(ModelSpec(name="j", provider="echo", model_id="none"))
    model.script(build_multi_judge_prompt(q, a), _multi_response())
    grader = MultiDimensionalGrader(model, compare_exact=True)
    result = grader.grade(q, a)
    assert result.verdict is not None
    assert result.correct == result.verdict.curriculum_correct
    assert result.score == result.verdict.curriculum_score
    assert "alignment=" in (result.feedback or "")


def test_escalating_uses_multi_judge_on_paraphrase() -> None:
    q = Question(id="q1", prompt="What is plasma?", expected="dynamic matter and fields", topic="plasma")
    a = Answer(question_id="q1", text="free ions and magnetic fields", confidence=0.5)
    model = EchoModel(ModelSpec(name="j", provider="echo", model_id="none"))
    model.script(build_multi_judge_prompt(q, a), _multi_response())
    fallback = MultiDimensionalGrader(model, compare_exact=True)
    grader = EscalatingGrader(ExactMatchGrader("contains"), fallback, escalate_paraphrases_only=True)
    result = grader.grade(q, a)
    assert result.verdict is not None
    assert result.correct


def test_disagreement_log_collects_mismatch() -> None:
    q = Question(id="q1", prompt="What is plasma?", expected="dynamic matter and fields", topic="plasma")
    a = Answer(question_id="q1", text="free ions", confidence=0.5)
    model = EchoModel(ModelSpec(name="j", provider="echo", model_id="none"))
    model.script(build_multi_judge_prompt(q, a), _multi_response())
    grader = MultiDimensionalGrader(model, compare_exact=True)
    result = grader.grade(q, a)
    exam = ExamResult(exam_id="exam-1", student_id="s1", results=(result,))
    rows = collect_disagreements(exam)
    assert rows
    assert rows[0]["curriculum_correct"] is True
    assert rows[0]["exact_match"] is False


def test_append_disagreements_jsonl(tmp_path: Path) -> None:
    q = Question(id="q1", prompt="p?", expected="a", topic="t")
    a = Answer(question_id="q1", text="b", confidence=0.5)
    model = EchoModel(ModelSpec(name="j", provider="echo", model_id="none"))
    model.script(build_multi_judge_prompt(q, a), _multi_response())
    result = MultiDimensionalGrader(model, compare_exact=True).grade(q, a)
    exam = ExamResult(exam_id="exam-1", student_id="s1", results=(result,))
    path = tmp_path / "review.jsonl"
    count = append_disagreements(path, exam)
    assert count == 1
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["exam_id"] == "exam-1"
    assert "curriculum_reason" in row
