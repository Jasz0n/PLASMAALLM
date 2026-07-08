"""Planner value objects: signals in, roadmap out.

A :class:`TopicSignal` bundles everything the planner may weigh for one
topic. Signals are assembled from the teacher's knowledge state plus a
topic catalog (see ``signals.py``); Phase 5's knowledge graph will
become the catalog's natural source without changing these types.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from allm.teacher.types import LearningGoal


class TopicSignal(BaseModel):
    """Everything known about one topic for one student."""

    model_config = ConfigDict(frozen=True)

    topic: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    dependencies: tuple[str, ...] = ()
    observations: int = Field(default=0, ge=0)
    forgetting_risk: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def weakness(self) -> float:
        """1 - confidence; an unexamined topic counts as fully weak."""
        return 1.0 - (self.confidence or 0.0)

    @property
    def novelty(self) -> float:
        """Decays with how often the topic has been examined."""
        return 1.0 / (1.0 + self.observations)


class RoadmapItem(BaseModel):
    """One prioritised entry in the learning roadmap."""

    model_config = ConfigDict(frozen=True)

    rank: int
    topic: str
    need: float
    weakness: float
    importance: float
    curiosity: float
    novelty: float
    forgetting_risk: float = 0.0
    blocked_by: tuple[str, ...] = ()
    reason: str


class Roadmap(BaseModel):
    """Ordered learning plan for one student."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    items: tuple[RoadmapItem, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_goals(self, max_goals: int | None = None) -> list[LearningGoal]:
        """Convert the top roadmap entries into teacher-style goals.

        Items with zero need (fully mastered, or a zero factor) are
        skipped — a goal nobody needs is noise. ``priority`` is the
        item's need clamped to [0, 1] (need is a product of [0, 1]
        factors, so it already lies there).
        """
        worthwhile = [item for item in self.items if item.need > 0.0]
        chosen = worthwhile if max_goals is None else worthwhile[:max_goals]
        return [
            LearningGoal(
                student_id=self.student_id,
                topic=item.topic,
                priority=round(min(1.0, max(0.0, item.need)), 4),
                reason=item.reason,
            )
            for item in chosen
        ]
