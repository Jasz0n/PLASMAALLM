"""Debate value objects."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from allm.data.base import Sample
from allm.exam.base import Answer, Question


class Position(BaseModel):
    """One student's stance in a debate."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    answer: Answer
    correct: bool | None = None  # None when no ground truth was available


class Cluster(BaseModel):
    """Students who gave (effectively) the same answer."""

    model_config = ConfigDict(frozen=True)

    answer_text: str
    members: tuple[str, ...]
    total_confidence: float

    @property
    def size(self) -> int:
        return len(self.members)


class DebateResult(BaseModel):
    """Outcome of one debated question."""

    model_config = ConfigDict(frozen=True)

    question: Question
    positions: tuple[Position, ...]
    clusters: tuple[Cluster, ...]
    disagreement: float = Field(ge=0.0, le=1.0)
    verdict: str
    unresolved: bool
    debated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_learning_sample(self) -> Sample:
        """Turn this debate into a study/research task.

        With ground truth the sample is trainable; without it the
        target is ``None`` — an open research task in Plan.md's sense.
        """
        return Sample(
            id=f"debate-{self.question.id}",
            input=self.question.prompt,
            target=self.question.expected,
            metadata={
                "topic": self.question.topic,
                "origin": "debate",
                "disagreement": self.disagreement,
                "verdict": self.verdict,
            },
        )
