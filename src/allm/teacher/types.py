"""Teacher value objects: configuration, goals, progress reports."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class TeacherConfig(BaseModel):
    """Tunable behaviour of the teacher.

    ``confidence_smoothing`` is the EMA weight of the newest exam score
    when updating a topic's confidence: 1.0 means "latest exam is the
    whole truth", lower values give students a memory of past
    performance. 0.5 is a neutral default for experiments.
    """

    model_config = ConfigDict(frozen=True)

    confidence_smoothing: float = Field(default=0.5, gt=0.0, le=1.0)
    weakness_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_goals: int = Field(default=3, ge=1)


class LearningGoal(BaseModel):
    """One assigned learning target for a student.

    Phase 2 priorities come straight from weakness (1 - confidence);
    Phase 4's planner replaces this with the full
    weakness x importance x curiosity x novelty scoring from Plan.md.
    """

    model_config = ConfigDict(frozen=True)

    student_id: str
    topic: str
    priority: float = Field(ge=0.0, le=1.0)
    reason: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TopicProgress(BaseModel):
    """Confidence trajectory for one (student, topic) pair."""

    model_config = ConfigDict(frozen=True)

    topic: str
    first: float
    latest: float
    observations: int

    @property
    def delta(self) -> float:
        return self.latest - self.first


class ProgressReport(BaseModel):
    """How a student has developed across all recorded exams."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    exams_taken: int
    mean_score: float
    topics: tuple[TopicProgress, ...]

    def improving(self) -> list[TopicProgress]:
        return [t for t in self.topics if t.delta > 0]

    def regressing(self) -> list[TopicProgress]:
        return [t for t in self.topics if t.delta < 0]
