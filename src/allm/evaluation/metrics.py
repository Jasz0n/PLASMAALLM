"""Success metrics from Plan.md, computed from recorded history.

Everything here is derived — no metric has its own storage, so metrics
can never drift from the ground truth in state and memory. Benchmarks
are deliberately absent: Plan.md measures improvement, transfer and
self-correction, not leaderboard scores.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from allm.memory.episodic import EpisodicMemory
from allm.teacher.state import KnowledgeState


class StudentEvaluation(BaseModel):
    """Aggregated Plan.md metrics for one student."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    improvement_per_topic: dict[str, float]
    learning_speed: float
    mastery: float
    self_correction_rate: float | None


def improvement_per_topic(state: KnowledgeState, student_id: str) -> dict[str, float]:
    """Confidence delta (latest - first) per examined topic."""
    deltas = {}
    for topic in state.topics(student_id):
        history = state.confidence_history(student_id, topic)
        if history:
            deltas[topic] = history[-1][1] - history[0][1]
    return deltas


def learning_speed(state: KnowledgeState, student_id: str) -> float:
    """Mean confidence gained per belief revision, across topics.

    0.0 for a student with no history; negative means regression.
    """
    total_gain = total_steps = 0.0
    for topic in state.topics(student_id):
        history = state.confidence_history(student_id, topic)
        if len(history) > 1:
            total_gain += history[-1][1] - history[0][1]
            total_steps += len(history) - 1
    return total_gain / total_steps if total_steps else 0.0


def mastery(state: KnowledgeState, student_id: str, threshold: float = 0.8) -> float:
    """Fraction of examined topics at or above the confidence threshold."""
    topics = state.topics(student_id)
    if not topics:
        return 0.0
    strong = sum(
        1 for t in topics if (state.confidence(student_id, t) or 0.0) >= threshold
    )
    return strong / len(topics)


def self_correction_rate(memory: EpisodicMemory, actor: str) -> float | None:
    """Of the questions this actor ever failed, the fraction it later
    answered correctly. Questions are matched by prompt, because
    question *ids* are exam-specific and never recur across exams.

    ``None`` when there are no failures yet — "no data" and "never
    corrects itself" must not look the same.
    """
    episodes = memory.recall(actor=actor)
    first_failure: dict[str, int] = {}
    corrected: set[str] = set()
    for index, episode in enumerate(episodes):
        question = str(episode.detail.get("prompt") or episode.summary)
        if episode.kind == "failure" and question not in first_failure:
            first_failure[question] = index
        elif episode.kind == "success" and question in first_failure:
            corrected.add(question)
    if not first_failure:
        return None
    return len(corrected) / len(first_failure)


def evaluate_student(
    state: KnowledgeState, memory: EpisodicMemory, student_id: str
) -> StudentEvaluation:
    """All Plan.md metrics for one student in one report."""
    return StudentEvaluation(
        student_id=student_id,
        improvement_per_topic=improvement_per_topic(state, student_id),
        learning_speed=learning_speed(state, student_id),
        mastery=mastery(state, student_id),
        self_correction_rate=self_correction_rate(memory, student_id),
    )
