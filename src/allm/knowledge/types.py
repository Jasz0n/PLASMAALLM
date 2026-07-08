"""Knowledge graph value objects.

A :class:`Concept` is the current snapshot of one node; its evolution
lives in the versioned record store (the graph never edits in place).
Plan.md requires every concept to know: prerequisites, related
concepts, confidence, evidence, source, and when it was learned —
plus usefulness, which the planner consumes as importance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Evidence(BaseModel):
    """One piece of support for (or against) a concept."""

    model_config = ConfigDict(frozen=True)

    source: str
    detail: str | None = None
    supports: bool = True
    recorded_at: datetime = Field(default_factory=_utcnow)


class Concept(BaseModel):
    """One node of the knowledge graph (current snapshot)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    prerequisites: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    usefulness: float = Field(default=0.5, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: tuple[Evidence, ...] = ()
    source: str = "unknown"
    status: Literal["active", "retracted"] = "active"
    learned_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
