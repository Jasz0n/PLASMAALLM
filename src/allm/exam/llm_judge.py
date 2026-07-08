"""LLM-as-judge grading for open-ended and reasoning answers.

The judge model receives the question, expected answer (when present)
and the student's answer, then returns a structured verdict. Disagreement
with :class:`~allm.exam.grading.ExactMatchGrader` is surfaced in
feedback so downstream metrics can treat it as a signal.
"""

from __future__ import annotations

import re

from allm.core.logging import get_logger
from allm.exam.base import Answer, Question, QuestionResult
from allm.exam.grading import ExactMatchGrader, graders
from allm.models.base import LanguageModel

logger = get_logger("exam.llm_judge")

_VERDICT = re.compile(r"^\s*VERDICT:\s*(correct|incorrect)\s*$", re.IGNORECASE | re.MULTILINE)
_SCORE = re.compile(r"^\s*SCORE:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)
_REASON = re.compile(r"^\s*REASON:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def build_judge_prompt(question: Question, answer: Answer) -> str:
    """The exact rubric prompt sent to the judge model."""
    lines = [
        "You are an exam grader. Decide if the student's answer is correct.",
        "Accept paraphrases and minor formatting differences when the meaning matches.",
        f"Question: {question.prompt}",
    ]
    if question.expected is not None:
        lines.append(f"Reference answer: {question.expected}")
    lines.extend(
        [
            f"Student answer: {answer.text}",
            "",
            "Reply with exactly these three lines and nothing else:",
            "VERDICT: correct|incorrect",
            "SCORE: 0.0-1.0",
            "REASON: one short sentence",
        ]
    )
    return "\n".join(lines)


def parse_judge_response(text: str) -> tuple[bool, float, str | None]:
    """Parse judge output into (correct, score, reason)."""
    verdict = _VERDICT.search(text)
    score_match = _SCORE.search(text)
    reason_match = _REASON.search(text)
    correct = verdict is not None and verdict.group(1).lower() == "correct"
    if score_match is not None:
        score = max(0.0, min(1.0, float(score_match.group(1))))
    else:
        score = 1.0 if correct else 0.0
    reason = reason_match.group(1).strip() if reason_match else None
    return correct, score, reason


@graders.register("llm_judge")
class LLMJudgeGrader:
    """Grades answers with a language model acting as rubric judge."""

    def __init__(
        self,
        model: LanguageModel,
        *,
        compare_exact: bool = True,
    ) -> None:
        self._model = model
        self._exact = ExactMatchGrader("contains") if compare_exact else None

    def grade(self, question: Question, answer: Answer) -> QuestionResult:
        raw = self._model.generate(build_judge_prompt(question, answer))
        correct, score, reason = parse_judge_response(raw)
        feedback = reason
        if self._exact is not None and question.expected is not None:
            exact = self._exact.grade(question, answer)
            if exact.correct != correct:
                logger.info(
                    "judge disagrees with exact_match on %s: judge=%s exact=%s",
                    question.id,
                    correct,
                    exact.correct,
                )
                tag = "judge/exact disagree"
                feedback = f"{tag}: {reason}" if reason else tag
        return QuestionResult(
            question=question,
            answer=answer,
            score=score,
            correct=correct,
            feedback=feedback,
        )
