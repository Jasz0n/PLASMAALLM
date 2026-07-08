"""Maintenance budget optimization — maximize expected KS gain per review slot (M42)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

from allm.collector.pool import SamplePool
from allm.data.base import Sample
from allm.planner.decay_prediction import decay_urgency, proactive_review_topics
from allm.planner.dependency_risk import dependency_boost
from allm.planner.forgetting_risk import review_topics_from_roadmap
from allm.planner.retrieval_strength import retrieval_strength
from allm.teacher.state import KnowledgeState

if TYPE_CHECKING:
    from allm.knowledge.graph import KnowledgeGraph
    from allm.planner.types import RoadmapItem


def maintenance_optimizer_enabled() -> bool:
    return os.environ.get("ALLM_MAINTENANCE_OPTIMIZER", "1") == "1"


def expected_ks_gain(
    *,
    forgetting_risk: float,
    importance: float,
    retrieval: float | None = None,
    dependency: float = 0.0,
    decay: float = 0.0,
) -> float:
    """Estimated KS improvement per review slot for one topic."""
    recall_gap = 1.0 - (retrieval if retrieval is not None else 0.5)
    raw = (
        forgetting_risk * 0.40
        + recall_gap * 0.30
        + decay * 0.20
        + dependency * 0.10
    )
    return round(raw * importance, 6)


def rank_review_topics(
    items: Sequence["RoadmapItem"],
    *,
    state: KnowledgeState,
    student_id: str,
    graph: "KnowledgeGraph | None" = None,
    limit: int = 6,
) -> list[tuple[str, float]]:
    """Rank topics by expected KS gain from one review slot."""
    current_risks = {item.topic: item.forgetting_risk for item in items}
    proactive = proactive_review_topics(
        state,
        student_id,
        [item.topic for item in items],
        current_risks=current_risks,
        limit=limit * 2,
    )
    reactive = review_topics_from_roadmap(items, limit=limit * 2)
    candidate_topics = list(dict.fromkeys([*reactive, *proactive]))
    if not candidate_topics:
        candidate_topics = [item.topic for item in items]

    by_topic = {item.topic: item for item in items}
    ranked: list[tuple[str, float]] = []
    for topic in candidate_topics:
        item = by_topic.get(topic)
        importance = item.importance if item is not None else 0.5
        risk = item.forgetting_risk if item is not None else current_risks.get(topic, 0.0)
        retrieval = retrieval_strength(state, student_id, topic)
        dependency = dependency_boost(graph, topic)
        decay = decay_urgency(state, student_id, topic)
        gain = expected_ks_gain(
            forgetting_risk=risk,
            importance=importance,
            retrieval=retrieval,
            dependency=dependency,
            decay=decay,
        )
        ranked.append((topic, gain))
    ranked.sort(key=lambda row: -row[1])
    return ranked[:limit]


def optimized_review_topics(
    items: Sequence["RoadmapItem"],
    *,
    state: KnowledgeState,
    student_id: str,
    graph: "KnowledgeGraph | None" = None,
    limit: int = 6,
) -> list[str]:
    """Review targets ordered by expected KS gain."""
    if not maintenance_optimizer_enabled():
        return review_topics_from_roadmap(items, limit=limit)
    return [topic for topic, _ in rank_review_topics(
        items,
        state=state,
        student_id=student_id,
        graph=graph,
        limit=limit,
    )]


def collect_prioritized_review_samples(
    pool: SamplePool,
    ranked_topics: list[str],
    *,
    limit: int,
    kinds: Sequence[str] | None = None,
) -> list[Sample]:
    """Fill review slots from highest-value topics first."""
    if limit <= 0 or not ranked_topics:
        return []
    collected: list[Sample] = []
    seen: set[str] = set()
    per_topic = max(1, limit // max(1, len(ranked_topics)))
    for topic in ranked_topics:
        rows = pool.collect(topics=[topic], limit=per_topic, kinds=kinds)
        for row in rows:
            if row.id in seen:
                continue
            seen.add(row.id)
            collected.append(row)
            if len(collected) >= limit:
                return collected[:limit]
    if len(collected) < limit:
        extras = pool.collect(topics=ranked_topics, limit=limit - len(collected), kinds=kinds)
        for row in extras:
            if row.id in seen:
                continue
            seen.add(row.id)
            collected.append(row)
            if len(collected) >= limit:
                break
    return collected[:limit]
