"""Dependency-weighted forgetting risk from the knowledge graph (M41)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from allm.knowledge.graph import KnowledgeGraph


def dependency_risk_enabled() -> bool:
    return os.environ.get("ALLM_DEPENDENCY_RISK", "1") == "1"


def dependency_boost(graph: "KnowledgeGraph | None", topic: str) -> float:
    """Raise maintenance priority when dependents rely on this topic."""
    if not dependency_risk_enabled() or graph is None:
        return 0.0
    dependents = graph.dependents_of(topic)
    if not dependents:
        return 0.0
    per_dependent = float(os.environ.get("ALLM_DEPENDENCY_RISK_PER_CHILD", "0.04"))
    cap = float(os.environ.get("ALLM_DEPENDENCY_RISK_CAP", "0.35"))
    return round(min(cap, per_dependent * len(dependents)), 4)


def downstream_topics(graph: "KnowledgeGraph | None", topic: str) -> tuple[str, ...]:
    """Direct dependents for maintenance propagation."""
    if graph is None:
        return ()
    return tuple(graph.dependents_of(topic))
