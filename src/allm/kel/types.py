"""KEL value objects.

A :class:`KELReport` is one measurement of the knowledge system at one
point in time; metric fields are ``None`` when their data source has no
data yet (``None`` = "cannot measure", never conflated with 0.0 =
"measured as zero"). :class:`GraphSnapshot` is the frozen structural
fingerprint used by the stability metric; :class:`Finding` is one
detected failure mode from KEL.md section 9.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FailureMode = Literal[
    "false_compression",
    "dead_knowledge_growth",
    "conflict_accumulation",
    "static_illusion",
    "unstable_mastery",
    "research_gap",
    "research_saturation",
    "high_conflict_discovery",
    "unearned_confidence",
]


class KELConfig(BaseModel):
    """Thresholds and normalisation targets — fixed, documented, tunable.

    ``target_reuse`` normalises CRR for the composite score: a concept
    used that many times counts as fully "alive". The failure-mode
    thresholds implement KEL.md section 9 and are deliberately
    conservative defaults for experiments.
    """

    model_config = ConfigDict(frozen=True)

    target_reuse: float = Field(default=5.0, gt=0.0)
    high_rcr: float = 0.5
    low_crr: float = 1.0
    high_cd: float = 0.2
    low_cre: float = 0.5
    high_gst: float = 0.9
    high_missing_knowledge: float = 0.4
    high_research_saturation: float = 0.7
    high_conflict_discovery: float = 0.35
    low_ks: float = 0.70
    # Documents propose, evidence disposes (KEL.md 9.5): confidence above
    # this cap without a single evidence package is unearned.
    evidence_confidence_cap: float = 0.75


class GraphSnapshot(BaseModel):
    """Structural fingerprint of the graph at one moment.

    Nodes are active concept names; edges are (concept, kind, target)
    triples over prerequisites and relations. Two snapshots compare via
    Jaccard similarity of these sets.
    """

    model_config = ConfigDict(frozen=True)

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    taken_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KELReport(BaseModel):
    """One full KEL measurement (KEL.md section 3 + 6)."""

    model_config = ConfigDict(frozen=True)

    rcr: float | None = None  # redundancy collapse ratio
    cd: float | None = None  # conflict density
    gst: float | None = None  # graph stability vs previous snapshot
    crr: float | None = None  # mean downstream uses per concept
    lg: float | None = None  # learning gain (mean confidence delta)
    cre: float | None = None  # conflict resolution efficiency
    egr: float | None = None  # evidence growth rate (foundation delta)
    ks: float | None = None  # knowledge stability across subsequent learning
    ghs: float | None = None  # composite graph health score
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Finding(BaseModel):
    """One detected failure mode."""

    model_config = ConfigDict(frozen=True)

    mode: FailureMode
    detail: str
