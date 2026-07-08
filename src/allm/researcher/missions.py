"""Research missions — persistent goals above per-cycle plans."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.kdp.types import content_hash
from allm.storage.base import RecordStore

NAMESPACE = "researcher_missions"
MissionStatus = Literal["open", "in_progress", "packaged", "closed"]


class ResearchMission(BaseModel):
    """A durable research goal with tasks and lifecycle."""

    model_config = ConfigDict(frozen=True)

    id: str
    goal: str
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: MissionStatus = "open"
    target_topics: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    source: str = "researcher"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def build(
        cls,
        *,
        goal: str,
        target_topics: tuple[str, ...] = (),
        tasks: tuple[str, ...] = (),
        priority: float = 0.6,
    ) -> "ResearchMission":
        mission_id = "rmission_" + content_hash(goal, *target_topics)
        return cls(
            id=mission_id,
            goal=goal,
            priority=priority,
            target_topics=target_topics,
            tasks=tasks or ("discover sources", "build package", "verify", "recommend to teacher"),
        )


class MissionStore:
    """Append-only mission lifecycle over the versioned store."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def save(self, mission: ResearchMission, *, reason: str = "mission update") -> ResearchMission:
        self._store.put(
            NAMESPACE,
            mission.id,
            json.loads(mission.model_dump_json()),
            reason=reason,
        )
        return mission

    def get(self, mission_id: str) -> ResearchMission | None:
        record = self._store.get(NAMESPACE, mission_id)
        if record is None:
            return None
        return ResearchMission.model_validate(record.value)

    def active(self) -> list[ResearchMission]:
        rows: list[ResearchMission] = []
        for key in self._store.keys(NAMESPACE):
            record = self._store.get(NAMESPACE, key)
            if record is None:
                continue
            mission = ResearchMission.model_validate(record.value)
            if mission.status in {"open", "in_progress"}:
                rows.append(mission)
        return sorted(rows, key=lambda row: (-row.priority, row.id))

    def open_from_gap(
        self,
        *,
        parent: str,
        child: str,
        missing: str,
        priority: float = 0.7,
    ) -> ResearchMission:
        """Create a mission for a missing prerequisite between graph nodes."""
        goal = f"Bridge gap: {parent} → {missing} → {child}"
        mission = ResearchMission.build(
            goal=goal,
            target_topics=(missing,),
            tasks=(
                f"find evidence for prerequisite {missing!r}",
                f"connect {parent!r} to {child!r}",
                "build knowledge package",
                "notify teacher",
            ),
            priority=priority,
        )
        existing = self.get(mission.id)
        if existing is not None:
            return existing
        return self.save(mission, reason="opened from graph gap")
