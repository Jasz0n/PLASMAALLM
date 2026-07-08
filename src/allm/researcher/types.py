"""Researcher value objects — Knowledge Packages and recommendations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.kdp.types import content_hash
from allm.researcher.multimodal_types import DistilledVisualBrief, StudentVisualPackage, SyncedEvidence

KnowledgeTier = Literal["established", "emerging", "hypothesis"]
RecommendationKind = Literal["discovery", "maintenance", "remediation"]
ConflictStatus = Literal["agreement", "disagreement", "unknown"]
ProviderKind = Literal["workshop", "software", "book", "paper", "repository", "network", "video"]


class PackageConcept(BaseModel):
    """One concept inside a Knowledge Package."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    relationships: tuple[str, ...] = ()
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    knowledge_tier: KnowledgeTier = "emerging"


class PackageConflict(BaseModel):
    """Preserved disagreement between sources."""

    model_config = ConfigDict(frozen=True)

    concept: str
    status: ConflictStatus = "unknown"
    sources: tuple[str, ...] = ()
    detail: str = ""


class KnowledgePackage(BaseModel):
    """Verified knowledge ready for Teacher review — never given raw to students."""

    model_config = ConfigDict(frozen=True)

    id: str
    provider: str
    title: str
    concepts: tuple[PackageConcept, ...] = ()
    definitions: tuple[tuple[str, str], ...] = ()
    procedures: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    misconceptions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    provenance: str = ""
    source_refs: tuple[str, ...] = ()
    conflicts: tuple[PackageConflict, ...] = ()
    curriculum_topic: str | None = None
    multimodal_evidence: tuple[SyncedEvidence, ...] = ()
    distilled_visual_briefs: tuple[DistilledVisualBrief, ...] = ()
    student_visual_packages: tuple[StudentVisualPackage, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def build(cls, *, provider: str, title: str, **kwargs) -> "KnowledgePackage":
        """Construct with a deterministic content-derived id."""
        pkg_id = "kpkg_" + content_hash(provider, title, str(kwargs.get("provenance", "")))
        return cls(id=pkg_id, provider=provider, title=title, **kwargs)


class ProviderReputation(BaseModel):
    """Dynamic trust score for one knowledge provider."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    kind: ProviderKind
    accuracy: float = Field(default=0.5, ge=0.0, le=1.0)
    freshness: float = Field(default=0.5, ge=0.0, le=1.0)
    packages_submitted: int = Field(default=0, ge=0)
    packages_accepted: int = Field(default=0, ge=0)

    @property
    def score(self) -> float:
        """Combined reputation for source evaluation."""
        acceptance = (
            self.packages_accepted / self.packages_submitted
            if self.packages_submitted
            else 0.5
        )
        return round(0.4 * self.accuracy + 0.3 * acceptance + 0.3 * self.freshness, 4)


class ResearchRecommendation(BaseModel):
    """Topic the Researcher suggests the Teacher prioritize — not a student assignment."""

    model_config = ConfigDict(frozen=True)

    topic: str
    priority: float = Field(ge=0.0, le=1.0)
    reason: str
    package_id: str
    provider: str
    concept: str | None = None
    suggested_students: tuple[str, ...] = ()
    skip_students: tuple[str, ...] = ()
    debate_candidate: bool = False
    proposal_hint: str | None = None
    mission_id: str | None = None
    knowledge_tier: KnowledgeTier = "emerging"
    recommendation_kind: RecommendationKind = "discovery"
    recommended_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearcherReport(BaseModel):
    """Output of one Researcher cycle."""

    model_config = ConfigDict(frozen=True)

    packages: tuple[KnowledgePackage, ...] = ()
    recommendations: tuple[ResearchRecommendation, ...] = ()
    providers_evaluated: int = 0
    conflicts_detected: int = 0
    plan: object | None = None
    proposal_hints: tuple[str, ...] = ()
    capability_summary: tuple[tuple[str, int, str], ...] = ()
    strategy_hints: object | None = None
    curiosity_signals: tuple = ()
    graph_gaps: tuple = ()
    active_missions: tuple = ()
    multimodal_synced: tuple = ()
    cross_source_report: object | None = None
