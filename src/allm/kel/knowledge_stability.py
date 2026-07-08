"""Knowledge Stability (KS) — retention across subsequent learning (M39)."""

from __future__ import annotations

from allm.teacher.state import KnowledgeState
from allm.trainer.forgetting import ForgettingReport


def topic_stability(state: KnowledgeState, student_id: str, topic: str) -> float | None:
    """Return current confidence relative to historical peak for one topic."""
    history = state.confidence_history(student_id, topic)
    if len(history) < 2:
        return None
    peak = max(confidence for _, confidence in history)
    current = history[-1][1]
    if peak <= 0:
        return None
    return round(min(1.0, current / peak), 4)


def knowledge_stability(
    state: KnowledgeState,
    student_id: str,
    topics: tuple[str, ...] | list[str] | None = None,
) -> float | None:
    """Mean topic stability — 1.0 means no regression from prior peaks."""
    topic_rows = list(topics) if topics else state.topics(student_id)
    scores: list[float] = []
    for topic in topic_rows:
        stability = topic_stability(state, student_id, topic)
        if stability is not None:
            scores.append(stability)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def ks_from_forgetting(reports: tuple[ForgettingReport, ...]) -> float | None:
    """Derive KS from forgetting watchdog probes when confidence history is thin."""
    if not reports:
        return None
    probed = 0
    stable = 0
    for report in reports:
        probed += len(report.probed_topics)
        stable += len(report.probed_topics) - len(report.regressions)
    if probed == 0:
        return None
    return round(stable / probed, 4)


def merge_ks(
    confidence_ks: float | None,
    forgetting_ks: float | None,
    retrieval_ks: float | None = None,
    mastery_ks: float | None = None,
    cross_topic_ks: float | None = None,
    debate_ks: float | None = None,
) -> float | None:
    """Combine confidence-history KS with probes, retrieval, and stability signals."""
    values = [
        value
        for value in (
            confidence_ks,
            forgetting_ks,
            retrieval_ks,
            mastery_ks,
            cross_topic_ks,
            debate_ks,
        )
        if value is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def retrieval_ks_from_state(
    state: KnowledgeState,
    student_id: str,
    topics: tuple[str, ...] | list[str],
) -> float | None:
    """Mean retrieval strength across topics — higher means more durable recall."""
    from allm.planner.retrieval_strength import retrieval_strength

    scores: list[float] = []
    for topic in topics:
        strength = retrieval_strength(state, student_id, topic)
        if strength is not None:
            scores.append(strength)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def mastery_stability_ks(
    state: KnowledgeState,
    student_id: str,
    topics: tuple[str, ...] | list[str],
) -> float | None:
    """Mean sustained-mastery score — distinguishes spikes from durable mastery."""
    from allm.planner.retrieval_strength import mastery_stability

    scores: list[float] = []
    for topic in topics:
        stability = mastery_stability(state, student_id, topic)
        if stability is not None:
            scores.append(stability)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def cross_topic_coherence(
    state: KnowledgeState,
    student_id: str,
    topics: tuple[str, ...] | list[str] | None = None,
) -> float | None:
    """1 minus normalized variance of per-topic stability — higher is more coherent."""
    topic_rows = list(topics) if topics else state.topics(student_id)
    stabilities: list[float] = []
    for topic in topic_rows:
        stability = topic_stability(state, student_id, topic)
        if stability is not None:
            stabilities.append(stability)
    if len(stabilities) < 2:
        return None
    mean = sum(stabilities) / len(stabilities)
    variance = sum((value - mean) ** 2 for value in stabilities) / len(stabilities)
    coherence = max(0.0, min(1.0, 1.0 - variance * 4.0))
    return round(coherence, 4)


def debate_consistency_ks(disagreement: float | None) -> float | None:
    """Higher when peer debate disagreement is low."""
    if disagreement is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - disagreement)), 4)


def degrading_topics(
    state: KnowledgeState,
    student_id: str,
    *,
    regression_threshold: float = 0.12,
) -> tuple[str, ...]:
    """Topics whose confidence fell materially below their historical peak."""
    rows: list[tuple[str, float]] = []
    for topic in state.topics(student_id):
        history = state.confidence_history(student_id, topic)
        if len(history) < 2:
            continue
        peak = max(confidence for _, confidence in history)
        current = history[-1][1]
        drop = peak - current
        if drop >= regression_threshold:
            rows.append((topic, drop))
    rows.sort(key=lambda row: -row[1])
    return tuple(topic for topic, _ in rows)
