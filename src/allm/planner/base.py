"""Planner interface."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from allm.core.registry import Registry
from allm.planner.types import Roadmap, TopicSignal


@runtime_checkable
class Planner(Protocol):
    """Turns per-topic signals into an ordered learning roadmap."""

    def plan(self, student_id: str, signals: Sequence[TopicSignal]) -> Roadmap: ...


planners: Registry[type] = Registry("planner")
