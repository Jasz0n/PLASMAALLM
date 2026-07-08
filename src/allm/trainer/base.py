"""Trainer interface: how students learn from samples.

Design decisions
----------------
- ``Trainer.train(student, samples)`` is deliberately agnostic about
  *how* learning happens. Phase 3 ships in-context learning (studying
  notes), which is real, fast and fully testable offline. Weight-level
  fine-tuning (LoRA via peft) is a planned second backend behind this
  same protocol — proposed, not yet implemented, so we never ship an
  untestable trainer.
- Trainers return a :class:`TrainingReport` so the learning loop can
  log what was studied and correlate it with later exam scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from allm.core.registry import Registry
from allm.data.base import Sample
from allm.students.model_student import ModelStudent


class TrainingReport(BaseModel):
    """What one training call did."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    method: str
    samples_used: int
    samples_skipped: int = 0
    adapter_id: str | None = None
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@runtime_checkable
class Trainer(Protocol):
    """Teaches a student from labelled samples."""

    def train(self, student: ModelStudent, samples: Iterable[Sample]) -> TrainingReport: ...


trainers: Registry[type] = Registry("trainer")
