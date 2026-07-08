"""In-context trainer: learning as note-taking.

Feeds labelled samples into the student's study memory. Samples
without a target cannot teach anything and are counted as skipped
rather than silently dropped.
"""

from __future__ import annotations

from typing import Iterable

from allm.core.logging import get_logger
from allm.data.base import Sample
from allm.students.model_student import ModelStudent
from allm.trainer.base import TrainingReport, trainers

logger = get_logger("trainer.in_context")


@trainers.register("in_context")
class InContextTrainer:
    """Turns samples into studied notes on the student."""

    def train(self, student: ModelStudent, samples: Iterable[Sample]) -> TrainingReport:
        used = skipped = 0
        for sample in samples:
            if sample.target is None:
                skipped += 1
                continue
            pinned = bool(sample.metadata.get("pin"))
            student.study(sample.input, sample.target, pinned=pinned)
            used += 1
        logger.info(
            "%s studied %d sample(s), skipped %d unlabelled",
            student.student_id,
            used,
            skipped,
        )
        return TrainingReport(
            student_id=student.student_id,
            method="in_context",
            samples_used=used,
            samples_skipped=skipped,
        )
