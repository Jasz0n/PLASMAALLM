"""Capability-based curriculum progression (M39)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from allm.loop.learning_loop import IterationReport
from allm.loop.strategy import LearningStrategy, advance_strategy


@dataclass(frozen=True)
class CapabilityRequirement:
    """What must be true before advancing past a strategy phase."""

    min_peak: float
    min_ks: float | None = None
    min_stable_iterations: int = 1


DEFAULT_REQUIREMENTS: dict[LearningStrategy, CapabilityRequirement] = {
    LearningStrategy.DEFINITIONS: CapabilityRequirement(min_peak=0.20, min_ks=None),
    LearningStrategy.RELATIONS: CapabilityRequirement(min_peak=0.28, min_ks=0.70, min_stable_iterations=2),
    LearningStrategy.REASONING: CapabilityRequirement(min_peak=0.32, min_ks=0.75, min_stable_iterations=2),
    LearningStrategy.RESEARCH: CapabilityRequirement(min_peak=0.35, min_ks=0.80, min_stable_iterations=3),
}


def capability_progression_enabled() -> bool:
    """Whether strategy advance uses KS + capability checks."""
    return os.environ.get("ALLM_KS_PROGRESSION", "1") == "1"


def ks_advance_threshold() -> float:
    return float(os.environ.get("ALLM_KS_ADVANCE_THRESHOLD", "0.70"))


def recent_scores(reports: list[IterationReport], window: int = 3) -> list[float]:
    rows = reports[-window:] if window > 0 else reports
    return [row.students[0].score_after for row in rows if row.students]


def capability_allows_advance(
    current: LearningStrategy,
    reports: list[IterationReport],
    *,
    ks: float | None,
) -> tuple[bool, str]:
    """Return whether the student demonstrates stable capability for the next phase."""
    if not capability_progression_enabled():
        return True, ""

    next_strategy = advance_strategy(current)
    if next_strategy is None:
        return False, "already at research"

    requirement = DEFAULT_REQUIREMENTS.get(current)
    if requirement is None:
        return True, ""

    scores = recent_scores(reports)
    if not scores:
        return False, "no exam history"

    peak = max(scores)
    if peak < requirement.min_peak:
        return False, f"peak {peak:.2f} < {requirement.min_peak:.2f}"

    if requirement.min_ks is not None:
        threshold = max(requirement.min_ks, ks_advance_threshold())
        if ks is None or ks < threshold:
            return False, f"KS {ks} < {threshold:.2f}"

    if requirement.min_stable_iterations > 1 and len(scores) < requirement.min_stable_iterations:
        return False, "insufficient stable iterations"

    if requirement.min_stable_iterations > 1:
        tail = scores[-requirement.min_stable_iterations :]
        rolling = sum(tail) / len(tail)
        if rolling < requirement.min_peak * 0.85:
            return False, f"rolling {rolling:.2f} unstable"

    return True, ""
