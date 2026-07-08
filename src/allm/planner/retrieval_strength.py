"""Retrieval strength and mastery stability from exam history (M41)."""

from __future__ import annotations

import os

from allm.teacher.state import KnowledgeState


def retrieval_exam_window() -> int:
    return int(os.environ.get("ALLM_RETRIEVAL_EXAM_WINDOW", "6"))


def retrieval_strength(
    state: KnowledgeState,
    student_id: str,
    topic: str,
) -> float | None:
    """Recall success rate for a topic over recent exams (0=weak, 1=strong)."""
    exams = state.exam_results(student_id)[-retrieval_exam_window() :]
    correct = 0
    total = 0
    for exam in exams:
        for result in exam.results:
            if result.question.topic != topic:
                continue
            total += 1
            if result.correct:
                correct += 1
    if total == 0:
        return None
    return round(correct / total, 4)


def mastery_stability(
    state: KnowledgeState,
    student_id: str,
    topic: str,
    *,
    min_observations: int = 3,
) -> float | None:
    """Distinguish sustained mastery from a recent confidence spike."""
    history = state.confidence_history(student_id, topic)
    if len(history) < min_observations:
        return None
    peak = max(confidence for _, confidence in history)
    if peak <= 0:
        return None
    recent = [confidence for _, confidence in history[-min_observations:]]
    sustained = min(recent) / peak
    span_bonus = min(1.0, len(history) / 8.0)
    return round(min(1.0, sustained * 0.85 + span_bonus * 0.15), 4)


def retrieval_risk(
    state: KnowledgeState,
    student_id: str,
    topic: str,
) -> float | None:
    """Forgetting risk derived from failed recalls (1 - retrieval strength)."""
    strength = retrieval_strength(state, student_id, topic)
    if strength is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - strength)), 4)
