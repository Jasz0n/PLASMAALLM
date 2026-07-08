"""Forgetting-risk signals for KS-driven planning (M40–M41)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from allm.planner.dependency_risk import dependency_boost
from allm.planner.decay_prediction import decay_prediction_enabled, decay_urgency
from allm.planner.retrieval_strength import mastery_stability, retrieval_risk
from allm.teacher.state import KnowledgeState

if TYPE_CHECKING:
    from allm.knowledge.graph import KnowledgeGraph


def _topic_stability(state: KnowledgeState, student_id: str, topic: str) -> float | None:
    """Return current confidence relative to historical peak for one topic."""
    history = state.confidence_history(student_id, topic)
    if len(history) < 2:
        return None
    peak = max(confidence for _, confidence in history)
    current = history[-1][1]
    if peak <= 0:
        return None
    return round(min(1.0, current / peak), 4)


def forgetting_risk_weight() -> float:
    """Blend weight for maintenance need in the planner."""
    return float(os.environ.get("ALLM_FORGETTING_RISK_WEIGHT", "0.65"))


def ks_planner_enabled() -> bool:
    """Whether forgetting risk participates in need scoring."""
    return os.environ.get("ALLM_KS_PLANNER", "1") == "1"


def topic_forgetting_risk(
    state: KnowledgeState,
    student_id: str,
    topic: str,
    *,
    maintenance_topics: set[str] | frozenset[str] | None = None,
    global_ks: float | None = None,
    observations: int = 0,
    graph: "KnowledgeGraph | None" = None,
) -> float:
    """Estimate how likely a topic is to be forgotten (0=stable, 1=at risk)."""
    maintenance = maintenance_topics or frozenset()
    components: list[tuple[float, float]] = []

    if topic in maintenance:
        components.append((0.9, 0.35))

    stability = _topic_stability(state, student_id, topic)
    if stability is not None:
        components.append((max(0.0, min(1.0, 1.0 - stability)), 0.30))
    elif observations > 0:
        components.append((0.15, 0.10))

    recall = retrieval_risk(state, student_id, topic)
    if recall is not None:
        components.append((recall, 0.25))

    sustained = mastery_stability(state, student_id, topic)
    if sustained is not None:
        components.append((max(0.0, min(1.0, 1.0 - sustained)), 0.10))

    if not components:
        risk = 0.0
    else:
        weight_sum = sum(weight for _, weight in components)
        risk = sum(value * weight for value, weight in components) / weight_sum

    risk = min(1.0, risk + dependency_boost(graph, topic))

    if decay_prediction_enabled():
        urgency = decay_urgency(state, student_id, topic)
        if urgency > 0:
            risk = min(1.0, risk + urgency * 0.25)

    if global_ks is not None and global_ks < 0.70:
        risk = min(1.0, risk + (0.70 - global_ks) * 0.5)
    return round(risk, 4)


def review_topics_from_roadmap(
    items: tuple,
    *,
    min_risk: float | None = None,
    limit: int = 6,
) -> list[str]:
    """Topics the planner ranked as high forgetting-risk review targets."""
    threshold = min_risk
    if threshold is None:
        threshold = float(os.environ.get("ALLM_REVIEW_RISK_THRESHOLD", "0.35"))
    ranked = sorted(
        (item for item in items if getattr(item, "forgetting_risk", 0.0) >= threshold),
        key=lambda row: (-getattr(row, "forgetting_risk", 0.0), getattr(row, "rank", 0)),
    )
    rows: list[str] = []
    for item in ranked:
        if item.topic not in rows:
            rows.append(item.topic)
        if len(rows) >= limit:
            break
    return rows
