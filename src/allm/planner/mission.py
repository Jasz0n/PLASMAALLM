"""Mission-aware importance scaling for the planner."""

from __future__ import annotations

from allm.planner.types import TopicSignal
from allm.students.identity import StudentIdentity, domain_fit


def apply_mission_weights(
    signals: list[TopicSignal],
    identity: StudentIdentity,
    *,
    seed: int = 0,
) -> list[TopicSignal]:
    """Scale topic importance by how well each topic fits the student's mission."""
    weighted: list[TopicSignal] = []
    for signal in signals:
        multiplier, _reason = domain_fit(signal.topic, identity, seed=seed)
        weighted.append(
            signal.model_copy(
                update={"importance": round(min(1.0, signal.importance * multiplier), 6)}
            )
        )
    return weighted
