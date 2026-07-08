"""Evidence Package value objects (see smallVision.md).

An :class:`EvidencePackage` is one human contribution treated as
structured evidence, not "a post": the claim under investigation, the
artifacts behind it, the conditions it ran under, how to reproduce it,
and what happened. Packages are immutable and content-addressed; the
platform stores the actual files and passes references (URIs + hashes).

The ecosystem principle these types encode:

    Knowledge is earned through transparent evidence, not authority.
    Confidence increases through reproducible results, not popularity.
    Every conclusion remains traceable to the observations behind it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.kdp.types import content_hash

Outcome = Literal["supported", "challenged", "inconclusive"]

PackageKind = Literal[
    "experiment",   # original hands-on result (prototype, measurement run)
    "replication",  # independent re-run of another package
    "paper",        # published analysis / literature
    "transcript",   # workshop, lecture, discussion record
    "debate",       # an ALLM debate outcome submitted as evidence
    "observation",  # field observation without controlled setup
]


class Artifact(BaseModel):
    """Reference to a supporting file (design, photo, dataset, code).

    The core never stores blobs; ``uri`` points into the platform's
    storage and ``sha256`` pins the exact bytes for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    uri: str
    media_type: str = "application/octet-stream"
    sha256: str | None = None


class EvidencePackage(BaseModel):
    """One contribution: claim + evidence + outcome, fully traceable."""

    model_config = ConfigDict(frozen=True)

    id: str
    claim: str
    concept: str
    contributor: str
    kind: PackageKind = "experiment"
    outcome: Outcome
    artifacts: tuple[Artifact, ...] = ()
    measurements: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    reproduction_steps: tuple[str, ...] = ()
    replicates: str | None = None  # id of the package this re-runs
    related_concepts: tuple[str, ...] = ()
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def build(cls, **kwargs: Any) -> "EvidencePackage":
        """Construct with a deterministic content-derived id."""
        pkg_id = "pkg_" + content_hash(
            str(kwargs.get("claim", "")),
            str(kwargs.get("concept", "")),
            str(kwargs.get("contributor", "")),
            str(kwargs.get("kind", "experiment")),
            str(kwargs.get("outcome", "")),
            str(sorted(kwargs.get("measurements", {}).items())),
        )
        return cls(id=pkg_id, **kwargs)


class ConfidenceBreakdown(BaseModel):
    """Why a concept's evidential confidence is what it is.

    Returned alongside every confidence value so "nothing is hidden" —
    future researchers can inspect why confidence is high or low.
    """

    model_config = ConfigDict(frozen=True)

    concept: str
    value: float = Field(ge=0.0, le=1.0)
    support_weight: float
    challenge_weight: float
    inconclusive_weight: float
    contributors: int
    independent_replications: int
    packages: tuple[str, ...]  # every package id that entered the number
