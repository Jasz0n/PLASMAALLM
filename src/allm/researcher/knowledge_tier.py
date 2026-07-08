"""Knowledge tier classification for packages and recommendations."""

from __future__ import annotations

from allm.researcher.types import KnowledgeTier


def classify_knowledge_tier(
    *,
    in_graph: bool,
    graph_confidence: float | None,
    has_conflict: bool,
    package_confidence: float,
) -> KnowledgeTier:
    """Classify as established, emerging, or hypothesis."""
    if has_conflict or package_confidence < 0.4:
        return "hypothesis"
    if in_graph and graph_confidence is not None and graph_confidence >= 0.75:
        return "established"
    return "emerging"
