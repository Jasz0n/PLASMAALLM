"""Failure log: every mistake is kept, versioned, and reusable.

Plan.md principle 4: "Failure is valuable training data." Failures are
stored per student in the versioned record store (namespace
``failures``) and can be converted straight back into training samples,
which is how the measure -> learn loop closes in Phase 3.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from allm.data.base import Sample
from allm.exam.base import QuestionResult
from allm.storage.base import RecordStore

SEP = "::"


class FailureRecord(BaseModel):
    """One recorded mistake."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    question_id: str
    prompt: str
    expected: str | None
    given: str
    confidence: float
    topic: str
    feedback: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_sample(self) -> Sample:
        """Turn this failure into a labelled training sample."""
        if self.expected is None:
            raise ValueError(f"failure on {self.question_id} has no expected answer")
        return Sample(
            id=f"failure-{self.student_id}-{self.question_id}",
            input=self.prompt,
            target=self.expected,
            metadata={"topic": self.topic, "origin": "failure"},
        )


class FailureLog:
    """Per-student failure persistence over a :class:`RecordStore`."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def record(self, student_id: str, result: QuestionResult) -> FailureRecord:
        """Store one graded-incorrect answer."""
        failure = FailureRecord(
            student_id=student_id,
            question_id=result.question.id,
            prompt=result.question.prompt,
            expected=result.question.expected,
            given=result.answer.text,
            confidence=result.answer.confidence,
            topic=result.question.topic,
            feedback=result.feedback,
        )
        self._store.put(
            "failures",
            f"{student_id}{SEP}{result.question.id}",
            json.loads(failure.model_dump_json()),
            reason=f"failed {result.question.id} (confidence {failure.confidence:.2f})",
        )
        return failure

    def failures(self, student_id: str) -> list[FailureRecord]:
        """Latest failure per question for a student, oldest first."""
        records = []
        for key in self._store.keys("failures"):
            sid, _, _ = key.partition(SEP)
            if sid == student_id:
                stored = self._store.get("failures", key)
                records.append(FailureRecord.model_validate(stored.value))
        records.sort(key=lambda f: f.occurred_at)
        return records

    def training_samples(self, student_id: str) -> list[Sample]:
        """All failures that can be studied (have an expected answer)."""
        return [
            f.to_sample() for f in self.failures(student_id) if f.expected is not None
        ]
