"""L5 — Ecosystem analysis (student-topic matrix)."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)

TopicStatus = Literal["mastered", "struggling", "unseen"]


class StudentTopicMatrix(BaseModel):
    """Per-student topic mastery snapshot."""

    model_config = ConfigDict(frozen=True)

    rows: dict[str, dict[str, TopicStatus]] = Field(default_factory=dict)

    def status(self, student_id: str, topic: str) -> TopicStatus:
        return self.rows.get(student_id, {}).get(topic, "unseen")


def build_student_topic_matrix(
    state,
    student_ids: tuple[str, ...],
    topics: set[str],
    *,
    mastery_threshold: float,
) -> StudentTopicMatrix:
    """Build student × topic status matrix."""
    rows: dict[str, dict[str, TopicStatus]] = {}
    if state is None:
        return StudentTopicMatrix(rows=rows)

    for student_id in student_ids:
        row: dict[str, TopicStatus] = {}
        for topic in topics:
            confidence = state.confidence(student_id, topic)
            if confidence is None:
                row[topic] = "unseen"
            elif confidence >= mastery_threshold:
                row[topic] = "mastered"
            else:
                row[topic] = "struggling"
        rows[student_id] = row
    return StudentTopicMatrix(rows=rows)


class EcosystemAnalysisCapability:
    """L5 — analyze ecosystem state for planning and targeting."""

    level = 5
    name = "ecosystem.analyze"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        topics = {rec.topic for rec in pipeline.recommendations}
        topics.update(ctx.config.catalog_topics)
        matrix = build_student_topic_matrix(
            ctx.state,
            ctx.student_ids,
            topics,
            mastery_threshold=ctx.config.mastery_threshold,
        )
        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(topics),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"matrix": matrix},
        )
