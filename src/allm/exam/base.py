"""Exam vocabulary: questions, answers, exams, results.

Design decisions
----------------
- These types are the shared language between teacher and students, so
  they live in ``allm.exam`` (self-contained, models only) and both
  sides depend on them — never on each other's concrete classes.
- Everything is frozen and JSON-serialisable so exams and results can
  be stored versioned in the record store and attached to runs as
  artifacts.
- Phase 2 ships a dataset-backed generator; Phase 7 (exam engine) adds
  generative and cross-domain generators behind the same
  :class:`ExamGenerator` protocol and registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from allm.exam.verdicts import MultiDimensionalVerdict

from allm.core.registry import Registry

QuestionKind = Literal["factual", "reasoning", "coding", "cross_domain", "practice"]


class Question(BaseModel):
    """One exam question."""

    model_config = ConfigDict(frozen=True)

    id: str
    prompt: str
    expected: str | None = None
    topic: str = "general"
    kind: QuestionKind = "factual"


class Answer(BaseModel):
    """A student's answer, with self-reported confidence in [0, 1]."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str | None = None
    logprob_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    self_reported_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Exam(BaseModel):
    """An ordered set of questions."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    questions: tuple[Question, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def topics(self) -> list[str]:
        """Distinct topics covered, sorted."""
        return sorted({q.topic for q in self.questions})


class QuestionResult(BaseModel):
    """One graded answer."""

    model_config = ConfigDict(frozen=True)

    question: Question
    answer: Answer
    score: float = Field(ge=0.0, le=1.0)
    correct: bool
    feedback: str | None = None
    verdict: MultiDimensionalVerdict | None = None


class ExamResult(BaseModel):
    """A student's graded exam."""

    model_config = ConfigDict(frozen=True)

    exam_id: str
    student_id: str
    results: tuple[QuestionResult, ...]
    adapter_id: str | None = None
    taken_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def score(self) -> float:
        """Mean question score in [0, 1]; 0.0 for an empty exam."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def topic_scores(self) -> dict[str, float]:
        """Mean score per topic."""
        by_topic: dict[str, list[float]] = {}
        for result in self.results:
            by_topic.setdefault(result.question.topic, []).append(result.score)
        return {topic: sum(vals) / len(vals) for topic, vals in by_topic.items()}

    def failures(self) -> list[QuestionResult]:
        """Incorrect answers — valuable training data (Plan.md principle 4)."""
        return [r for r in self.results if not r.correct]

    def mean_curriculum_score(self) -> float | None:
        """Mean curriculum score when multi-judge verdicts are present."""
        from allm.exam.verdicts import mean_curriculum_score

        return mean_curriculum_score(self.results)

    def mean_alignment_score(self) -> float | None:
        from allm.exam.verdicts import mean_alignment_score

        return mean_alignment_score(self.results)

    def mean_evidence_score(self) -> float | None:
        from allm.exam.verdicts import mean_evidence_score

        return mean_evidence_score(self.results)


@runtime_checkable
class ExamGenerator(Protocol):
    """Produces exams, optionally restricted to given topics."""

    def generate(
        self,
        *,
        topics: Sequence[str] | None = None,
        num_questions: int = 10,
        seed: int | None = None,
    ) -> Exam: ...


exam_generators: Registry[type] = Registry("exam_generator")
