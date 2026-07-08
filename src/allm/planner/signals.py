"""Assembling planner signals from the teacher's knowledge state.

The catalog says what topics exist and how they relate (importance,
curiosity, prerequisites); the knowledge state says how the student is
actually doing (confidence, observation count). Phase 5's knowledge
graph will generate the catalog; until then it is declared by hand or
loaded from YAML.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from allm.planner.forgetting_risk import ks_planner_enabled, topic_forgetting_risk
from allm.planner.types import TopicSignal
from allm.students.identity import StudentIdentity
from allm.teacher.state import KnowledgeState


class TopicInfo(BaseModel):
    """Static description of one topic in the curriculum catalog."""

    model_config = ConfigDict(frozen=True)

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    dependencies: tuple[str, ...] = ()


def load_catalog(path: Path | str) -> dict[str, TopicInfo]:
    """Load a ``topic -> TopicInfo`` catalog from YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"catalog {path} must be a mapping of topic -> info")
    return {topic: TopicInfo.model_validate(info or {}) for topic, info in data.items()}


def build_signals(
    state: KnowledgeState,
    student_id: str,
    catalog: dict[str, TopicInfo],
    *,
    identity: StudentIdentity | None = None,
    mission_seed: int = 0,
    maintenance_topics: tuple[str, ...] | list[str] | None = None,
    global_ks: float | None = None,
    graph: object | None = None,
) -> list[TopicSignal]:
    """Merge catalog knowledge with the student's measured state.

    Topics appear if they are in the catalog *or* have been examined;
    examined-but-uncatalogued topics get neutral importance/curiosity.
    When ``identity`` is set, importance is scaled by mission fit.
    When ``global_ks`` is set, per-topic forgetting risk is computed for M40 planning.
    """
    from allm.planner.mission import apply_mission_weights

    maintenance = frozenset(maintenance_topics or ())
    topics = sorted(set(catalog) | set(state.topics(student_id)))
    signals = []
    for topic in topics:
        info = catalog.get(topic, TopicInfo())
        history = state.confidence_history(student_id, topic)
        risk = 0.0
        if ks_planner_enabled():
            risk = topic_forgetting_risk(
                state,
                student_id,
                topic,
                maintenance_topics=maintenance,
                global_ks=global_ks,
                observations=len(history),
                graph=graph,
            )
        signals.append(
            TopicSignal(
                topic=topic,
                confidence=state.confidence(student_id, topic),
                importance=info.importance,
                curiosity=info.curiosity,
                dependencies=info.dependencies,
                observations=len(history),
                forgetting_risk=risk,
            )
        )
    if identity is None:
        return signals
    return apply_mission_weights(signals, identity, seed=mission_seed)
