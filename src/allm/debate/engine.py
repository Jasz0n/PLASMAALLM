"""Debate engine: independent answers, measured disagreement.

Design decisions
----------------
- Students answer *independently* (no cross-talk in Phase 8): what we
  need first is an honest disagreement signal, not persuasion dynamics.
  Argument exchange can be layered on top of the same result types.
- Answers cluster by normalised text; disagreement is
  ``1 - largest_cluster / participants`` (0 = unanimous, -> 1 = total
  scatter).
- The verdict is the cluster with the highest *total confidence*, not
  the biggest head-count — two sure students outweigh three shruggers.
  With ground truth available each position is also graded, so the
  teacher can see when confident majorities are simply wrong.
- Debates at or above the disagreement threshold are ``unresolved`` and
  convert into learning tasks (Plan.md: disagreements become research
  tasks).
"""

from __future__ import annotations

from typing import Sequence

from allm.core.logging import get_logger
from allm.exam.base import Question
from allm.exam.grading import Grader, normalise
from allm.students.base import Student
from allm.debate.types import Cluster, DebateResult, Position

logger = get_logger("debate.engine")


class DebateEngine:
    """Runs one-question debates between students."""

    def __init__(
        self,
        grader: Grader | None = None,
        disagreement_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= disagreement_threshold <= 1.0:
            raise ValueError("disagreement_threshold must be in [0, 1]")
        self._grader = grader
        self._threshold = disagreement_threshold

    def debate(self, question: Question, students: Sequence[Student]) -> DebateResult:
        if len(students) < 2:
            raise ValueError("a debate needs at least two students")

        positions = tuple(
            Position(
                student_id=student.student_id,
                answer=(answer := student.solve(question)),
                correct=(
                    self._grader.grade(question, answer).correct
                    if self._grader is not None and question.expected is not None
                    else None
                ),
            )
            for student in students
        )
        clusters = self._cluster(positions)
        disagreement = 1.0 - max(c.size for c in clusters) / len(positions)
        verdict = max(clusters, key=lambda c: c.total_confidence).answer_text
        unresolved = disagreement >= self._threshold
        logger.info(
            "debate on %r: %d clusters, disagreement %.2f, %s",
            question.prompt,
            len(clusters),
            disagreement,
            "UNRESOLVED -> learning task" if unresolved else f"verdict {verdict!r}",
        )
        return DebateResult(
            question=question,
            positions=positions,
            clusters=clusters,
            disagreement=disagreement,
            verdict=verdict,
            unresolved=unresolved,
        )

    @staticmethod
    def _cluster(positions: tuple[Position, ...]) -> tuple[Cluster, ...]:
        grouped: dict[str, list[Position]] = {}
        for position in positions:
            grouped.setdefault(normalise(position.answer.text), []).append(position)
        clusters = [
            Cluster(
                answer_text=members[0].answer.text,
                members=tuple(p.student_id for p in members),
                total_confidence=sum(p.answer.confidence for p in members),
            )
            for members in grouped.values()
        ]
        clusters.sort(key=lambda c: (-c.total_confidence, c.answer_text))
        return tuple(clusters)
