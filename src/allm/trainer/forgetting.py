"""Forgetting watchdog: detect mastery regression after weight updates."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from allm.students.base import Student
from allm.teacher.state import KnowledgeState
from allm.teacher.teacher import Teacher


class ForgettingReport(BaseModel):
    """Topics that regressed after a training step."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    probed_topics: tuple[str, ...]
    regressions: dict[str, float] = Field(
        description="topic -> score drop (negative means regression)"
    )


class ForgettingWatchdog:
    """Re-examines previously mastered topics after fine-tuning."""

    def __init__(
        self,
        teacher: Teacher,
        *,
        mastery_threshold: float = 0.8,
        regression_threshold: float = 0.15,
        questions_per_topic: int = 2,
    ) -> None:
        self._teacher = teacher
        self._mastery = mastery_threshold
        self._regression = regression_threshold
        self._questions = questions_per_topic

    def mastered_topics(self, state: KnowledgeState, student_id: str) -> dict[str, float]:
        """Topics at or above mastery threshold before training."""
        result: dict[str, float] = {}
        for topic in state.topics(student_id):
            confidence = state.confidence(student_id, topic)
            if confidence is not None and confidence >= self._mastery:
                result[topic] = confidence
        return result

    def check(
        self,
        student: Student,
        mastered_before: dict[str, float],
        *,
        seed: int,
    ) -> ForgettingReport:
        """Probe mastered topics; flag regressions beyond the threshold."""
        regressions: dict[str, float] = {}
        for index, (topic, previous) in enumerate(sorted(mastered_before.items())):
            exam = self._teacher.create_exam(
                topics=[topic],
                num_questions=self._questions,
                seed=seed + index * 10_000,
            )
            if not exam.questions:
                continue
            result = self._teacher.evaluate(student, exam)
            topic_scores = result.topic_scores()
            current = topic_scores.get(topic, result.score)
            drop = current - previous
            if drop <= -self._regression:
                regressions[topic] = round(drop, 4)
        return ForgettingReport(
            student_id=student.student_id,
            probed_topics=tuple(sorted(mastered_before)),
            regressions=regressions,
        )
