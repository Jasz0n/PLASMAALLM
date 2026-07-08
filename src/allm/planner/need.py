"""The need-based planner from Plan.md's curiosity engine.

    Need = Weakness x Importance x Curiosity x Novelty
           + ForgettingRisk x Importance x weight   (M40)

Dependency handling (two rules, both curriculum common sense):

1. A topic whose prerequisites are not yet mastered is *blocked*: it is
   scheduled after all unblocked topics, whatever its need score.
   Prerequisites missing from the signals entirely count as unmet — we
   cannot assume knowledge we have never measured.
2. A blocked topic's urgency flows to its prerequisites: each unmet
   prerequisite's need is raised to at least ``blocked_boost`` times the
   blocked topic's need. Wanting quantum gravity makes general
   relativity urgent.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.planner.base import planners
from allm.planner.forgetting_risk import forgetting_risk_weight, ks_planner_enabled
from allm.planner.types import Roadmap, RoadmapItem, TopicSignal

logger = get_logger("planner.need")


class NeedPlannerConfig(BaseModel):
    """Tunables for the need-based planner."""

    model_config = ConfigDict(frozen=True)

    mastery_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    blocked_boost: float = Field(default=0.9, ge=0.0, le=1.0)


@planners.register("need")
class NeedPlanner:
    """Orders topics by need, prerequisites first."""

    def __init__(self, config: NeedPlannerConfig | None = None) -> None:
        self._config = config or NeedPlannerConfig()

    def plan(self, student_id: str, signals: Sequence[TopicSignal]) -> Roadmap:
        by_topic = {s.topic: s for s in signals}
        risk_weight = forgetting_risk_weight() if ks_planner_enabled() else 0.0
        need = {}
        for signal in signals:
            growth = signal.weakness * signal.importance * signal.curiosity * signal.novelty
            maintenance = signal.forgetting_risk * signal.importance * risk_weight
            need[signal.topic] = growth + maintenance
        blocked_by = {s.topic: self._unmet(s, by_topic) for s in signals}

        # Rule 2: urgency flows from blocked topics to their prerequisites.
        for signal in signals:
            for prerequisite in blocked_by[signal.topic]:
                if prerequisite in need:
                    boosted = self._config.blocked_boost * need[signal.topic]
                    if boosted > need[prerequisite]:
                        logger.debug(
                            "boosting %r to %.3f (unblocks %r)",
                            prerequisite,
                            boosted,
                            signal.topic,
                        )
                        need[prerequisite] = boosted

        # Rule 1: unblocked topics first, then by need, then stable by name.
        ordered = sorted(
            signals,
            key=lambda s: (bool(blocked_by[s.topic]), -need[s.topic], s.topic),
        )
        items = tuple(
            RoadmapItem(
                rank=rank,
                topic=s.topic,
                need=round(need[s.topic], 6),
                weakness=s.weakness,
                importance=s.importance,
                curiosity=s.curiosity,
                novelty=s.novelty,
                forgetting_risk=s.forgetting_risk,
                blocked_by=blocked_by[s.topic],
                reason=self._reason(s, need[s.topic], blocked_by[s.topic]),
            )
            for rank, s in enumerate(ordered, start=1)
        )
        return Roadmap(student_id=student_id, items=items)

    def _unmet(
        self, signal: TopicSignal, by_topic: dict[str, TopicSignal]
    ) -> tuple[str, ...]:
        unmet = []
        for dependency in signal.dependencies:
            known = by_topic.get(dependency)
            confidence = known.confidence if known is not None else None
            if confidence is None or confidence < self._config.mastery_threshold:
                unmet.append(dependency)
        return tuple(unmet)

    @staticmethod
    def _reason(signal: TopicSignal, need: float, blocked: tuple[str, ...]) -> str:
        parts = [
            f"need {need:.3f}",
            f"(weakness {signal.weakness:.2f} x importance {signal.importance:.2f}"
            f" x curiosity {signal.curiosity:.2f} x novelty {signal.novelty:.2f}"
            f" + forgetting {signal.forgetting_risk:.2f})",
        ]
        if blocked:
            parts.append(f"blocked by {', '.join(blocked)}")
        return " ".join(parts)
