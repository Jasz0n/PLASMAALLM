"""Merge Researcher recommendations into planner topic catalog."""

from __future__ import annotations

from allm.planner.signals import TopicInfo
from allm.researcher.types import ResearchRecommendation


def merge_research_recommendations(
    catalog: dict[str, TopicInfo],
    recommendations: list[ResearchRecommendation],
    *,
    boost: float = 0.85,
) -> dict[str, TopicInfo]:
    """Boost importance/curiosity for Researcher-prioritized topics."""
    if not recommendations:
        return catalog

    merged = dict(catalog)
    for rec in recommendations:
        existing = merged.get(rec.topic, TopicInfo())
        kind = getattr(rec, "recommendation_kind", "discovery")
        if kind == "maintenance":
            importance = min(1.0, max(existing.importance, rec.priority * boost))
            merged[rec.topic] = existing.model_copy(
                update={"importance": round(importance, 4)},
            )
            continue
        importance = min(1.0, max(existing.importance, rec.priority * boost))
        curiosity = min(1.0, max(existing.curiosity, rec.priority))
        merged[rec.topic] = existing.model_copy(
            update={"importance": round(importance, 4), "curiosity": round(curiosity, 4)}
        )
    return merged
