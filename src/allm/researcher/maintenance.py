"""Researcher maintenance recommendations when ecosystem knowledge degrades (M39)."""

from __future__ import annotations

import os

from allm.kel.knowledge_stability import degrading_topics
from allm.researcher.types import ResearchRecommendation
from allm.teacher.state import KnowledgeState


def build_maintenance_recommendations(
    state: KnowledgeState | None,
    student_ids: tuple[str, ...],
    *,
    regression_threshold: float | None = None,
    max_topics: int = 8,
) -> list[ResearchRecommendation]:
    """Emit maintenance recs for topics whose confidence regressed."""
    if state is None or not student_ids:
        return []
    threshold = regression_threshold
    if threshold is None:
        threshold = float(os.environ.get("ALLM_MAINTENANCE_REGRESSION", "0.12"))

    rows: list[ResearchRecommendation] = []
    seen: set[str] = set()
    for student_id in student_ids:
        for topic in degrading_topics(state, student_id, regression_threshold=threshold):
            if topic in seen:
                continue
            seen.add(topic)
            rows.append(
                ResearchRecommendation(
                    topic=topic,
                    priority=0.88,
                    reason=(
                        f"Researcher health: {topic} confidence dropped "
                        f">= {threshold:.2f} — schedule maintenance review"
                    ),
                    package_id=f"maintenance::{student_id}::{topic}",
                    provider="ecosystem-health",
                    concept=topic,
                    suggested_students=(student_id,),
                    recommendation_kind="maintenance",
                    knowledge_tier="established",
                )
            )
            if len(rows) >= max_topics:
                return rows
    return rows
