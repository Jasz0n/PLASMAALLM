"""Memory value objects.

An :class:`Episode` is one remembered event. Confidence history and
belief revisions are *not* duplicated here — they already live,
versioned, in the teacher state and knowledge graph; episodic memory
covers what happened (successes, failures, revisions, reasoning
traces, observations) as a queryable timeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EpisodeKind = Literal["success", "failure", "revision", "reasoning", "observation"]


class Episode(BaseModel):
    """One remembered event."""

    model_config = ConfigDict(frozen=True)

    id: str
    actor: str
    kind: EpisodeKind
    topic: str = "general"
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
