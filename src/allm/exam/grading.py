"""Grading: turning (question, answer) pairs into scores.

Phase 2 ships deterministic string grading, which is honest for the
factual, dataset-backed exams we generate now. LLM-judge grading for
open-ended reasoning arrives with Phase 7 as another ``graders`` entry;
the :class:`Grader` protocol will not change.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol, runtime_checkable

from allm.exam.escalation import might_be_paraphrase
from allm.exam.text import normalise
from allm.core.registry import Registry
from allm.exam.base import Answer, Question, QuestionResult


@runtime_checkable
class Grader(Protocol):
    """Scores one answer against one question."""

    def grade(self, question: Question, answer: Answer) -> QuestionResult: ...


graders: Registry[type] = Registry("grader")


from allm.exam.text import normalise
class CompositeGrader:
    """Routes questions to graders by ``question.kind``.

    The teacher takes exactly one grader; this is how mixed exams
    (factual + coding + ...) still grade correctly per question.
    """

    def __init__(self, default: Grader, by_kind: dict[str, Grader] | None = None) -> None:
        self._default = default
        self._by_kind = dict(by_kind or {})

    def grade(self, question: Question, answer: Answer) -> QuestionResult:
        grader = self._by_kind.get(question.kind, self._default)
        return grader.grade(question, answer)


@graders.register("escalating")
class EscalatingGrader:
    """Try a fast grader first; escalate to a judge when it fails.

    Exact-match is cheap and deterministic; the fallback (typically an
    LLM judge) runs only on incorrect or ungradeable answers.
    """

    def __init__(
        self,
        primary: Grader,
        fallback: Grader,
        *,
        escalate_paraphrases_only: bool = True,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._paraphrases_only = escalate_paraphrases_only

    def grade(self, question: Question, answer: Answer) -> QuestionResult:
        primary = self._primary.grade(question, answer)
        if primary.correct:
            return primary
        if question.expected is None:
            return self._fallback.grade(question, answer)
        if self._paraphrases_only and not might_be_paraphrase(
            answer.text, question.expected
        ):
            return primary
        fallback = self._fallback.grade(question, answer)
        if fallback.correct != primary.correct:
            feedback = fallback.feedback or "escalated to judge"
            return fallback.model_copy(update={"feedback": f"escalated: {feedback}"})
        return primary


@graders.register("exact_match")
class ExactMatchGrader:
    """Deterministic string grading against ``question.expected``.

    Modes:
        exact:    normalised answer == normalised expected
        contains: normalised expected appears in the normalised answer
                  (forgiving towards chatty model output)

    A question without an expected answer cannot be graded here and
    scores 0 with explanatory feedback — better loudly wrong than
    silently generous.
    """

    def __init__(self, mode: Literal["exact", "contains"] = "contains") -> None:
        self._mode = mode

    def grade(self, question: Question, answer: Answer) -> QuestionResult:
        if question.expected is None:
            return QuestionResult(
                question=question,
                answer=answer,
                score=0.0,
                correct=False,
                feedback="no expected answer; needs a judge grader (Phase 7)",
            )
        expected = normalise(question.expected)
        got = normalise(answer.text)
        correct = expected == got if self._mode == "exact" else expected in got
        return QuestionResult(
            question=question,
            answer=answer,
            score=1.0 if correct else 0.0,
            correct=correct,
            feedback=None if correct else f"expected {question.expected!r}",
        )
