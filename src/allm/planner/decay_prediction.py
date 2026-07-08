"""Proactive decay prediction for maintenance scheduling (M42)."""

from __future__ import annotations

import os

from allm.teacher.state import KnowledgeState


def decay_prediction_enabled() -> bool:
    return os.environ.get("ALLM_DECAY_PREDICTION", "1") == "1"


def decay_horizon() -> int:
    """Iterations ahead to extrapolate forgetting risk."""
    return max(1, int(os.environ.get("ALLM_DECAY_HORIZON", "2")))


def decay_risk_threshold() -> float:
    return float(os.environ.get("ALLM_DECAY_RISK_THRESHOLD", "0.35"))


def confidence_decay_rate(
    state: KnowledgeState,
    student_id: str,
    topic: str,
) -> float | None:
    """Per-observation confidence slope; negative means decaying."""
    history = state.confidence_history(student_id, topic)
    if len(history) < 2:
        return None
    start = history[0][1]
    end = history[-1][1]
    return round((end - start) / (len(history) - 1), 4)


def predicted_stability(
    state: KnowledgeState,
    student_id: str,
    topic: str,
    *,
    horizon: int | None = None,
) -> float | None:
    """Extrapolated confidence relative to peak after ``horizon`` iterations."""
    history = state.confidence_history(student_id, topic)
    if len(history) < 2:
        return None
    peak = max(confidence for _, confidence in history)
    if peak <= 0:
        return None
    rate = confidence_decay_rate(state, student_id, topic)
    if rate is None:
        return None
    steps = horizon if horizon is not None else decay_horizon()
    projected = max(0.0, min(1.0, history[-1][1] + rate * steps))
    return round(min(1.0, projected / peak), 4)


def decay_urgency(
    state: KnowledgeState,
    student_id: str,
    topic: str,
    *,
    horizon: int | None = None,
) -> float:
    """How urgently a topic needs review before predicted decay (0=stable, 1=imminent)."""
    if not decay_prediction_enabled():
        return 0.0
    stability = predicted_stability(state, student_id, topic, horizon=horizon)
    if stability is None:
        return 0.0
    threshold = decay_risk_threshold()
    predicted_risk = max(0.0, min(1.0, 1.0 - stability))
    if predicted_risk < threshold:
        return 0.0
    return round(min(1.0, (predicted_risk - threshold) / max(0.01, 1.0 - threshold)), 4)


def proactive_review_topics(
    state: KnowledgeState,
    student_id: str,
    topics: tuple[str, ...] | list[str],
    *,
    current_risks: dict[str, float] | None = None,
    limit: int = 6,
) -> list[str]:
    """Topics predicted to cross the decay threshold even if currently below it."""
    if not decay_prediction_enabled():
        return []
    risks = current_risks or {}
    threshold = decay_risk_threshold()
    ranked: list[tuple[str, float]] = []
    for topic in topics:
        current = risks.get(topic, 0.0)
        if current >= threshold:
            continue
        urgency = decay_urgency(state, student_id, topic)
        if urgency > 0:
            ranked.append((topic, urgency))
    ranked.sort(key=lambda row: -row[1])
    rows: list[str] = []
    for topic, _ in ranked:
        if topic not in rows:
            rows.append(topic)
        if len(rows) >= limit:
            break
    return rows
