"""Proposal value objects.

An :class:`ExperimentProposal` is the system asking the humans for
evidence: a concrete question about a concept, with the rationale that
produced it (a debate disagreement, a KDP conflict, a planner gap).
Lifecycle: ``open -> claimed -> resolved`` — resolution always happens
through evidence packages, never by decree.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProposalOrigin = Literal["debate", "conflict", "planner", "manual"]
ProposalStatus = Literal["open", "claimed", "resolved"]


class Resolution(BaseModel):
    """How a proposal was settled."""

    model_config = ConfigDict(frozen=True)

    outcome: Literal["supported", "challenged", "inconclusive"]
    package_ids: tuple[str, ...]
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentProposal(BaseModel):
    """One experiment the system wants a human to run."""

    model_config = ConfigDict(frozen=True)

    id: str
    concept: str
    question: str
    rationale: str
    origin: ProposalOrigin
    status: ProposalStatus = "open"
    claimed_by: str | None = None
    resolution: Resolution | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
