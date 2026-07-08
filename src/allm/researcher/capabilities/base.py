"""Capability protocols and shared pipeline state."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.types import KnowledgePackage, ResearchRecommendation
from allm.storage.base import RecordStore

METRICS_NAMESPACE = "researcher_capability_metrics"


class ResearcherPipelineConfig(BaseModel):
    """Static configuration for one Researcher cycle."""

    model_config = ConfigDict(frozen=True)

    workshop_dir: Path | None = None
    software_samples: Path | None = None
    repository_dir: Path | None = None
    repository_max_files: int | None = 48
    workshop_max_files: int | None = 3
    workshop_curriculum_topic: str = "kids-plasma"
    book_dir: Path | None = None
    book_max_files: int | None = 1
    book_max_pages: int = 32
    book_curriculum_topic: str = "kids-plasma"
    book_pdf_backend: str = "auto"
    enable_book_images: bool = False
    book_max_images: int = 24
    book_images_cache_dir: Path | None = None
    video_fixture_dir: Path | None = None
    video_dir: Path | None = None
    auto_generate_video_fixtures: bool = False
    enable_vision_captions: bool = False
    frames_cache_dir: Path | None = None
    vision_caption_backend: str = "stub"
    vision_ollama_model: str = "llava"
    enable_audio_analysis: bool = False
    audio_cache_dir: Path | None = None
    audio_clip_duration_sec: float = 2.5
    audio_analysis_backend: str = "auto"
    enable_frame_ocr: bool = False
    ocr_backend: str = "auto"
    ocr_ollama_model: str = "llava"
    enable_vision_analytics: bool = False
    vision_analytics_backend: str = "auto"
    enable_motion_tracking: bool = False
    motion_tracking_backend: str = "auto"
    motion_tracking_samples: int = 3
    motion_tracking_fps: float = 30.0
    enable_motion_continuity: bool = False
    motion_continuity_min_score: float = 0.35
    enable_object_identity: bool = False
    object_identity_min_score: float = 0.30
    enable_visual_distillation: bool = False
    visual_distillation_max_images: int = 3
    visual_distillation_max_questions: int = 5
    enable_visual_export: bool = False
    visual_export_auto_approve: bool = True
    visual_export_min_confidence: float = 0.7
    visual_export_max_images: int = 2
    visual_export_max_questions: int = 3
    visual_export_persist_approvals: bool = True
    enable_cross_source_verification: bool = True
    cross_source_min_overlap: float = 0.35
    enable_livekit: bool = False
    social_api_base_url: str | None = None
    livekit_fixture_path: Path | None = None
    livekit_stream_ids: tuple[str, ...] = ()
    livekit_topics: frozenset[str] = Field(default_factory=frozenset)
    livekit_researcher_identity: str = "plasma-researcher"
    livekit_observer_backend: str = "auto"
    livekit_cache_dir: Path | None = None
    livekit_capture_seconds: float = 3.0
    livekit_use_worker: bool = False
    enable_livekit_archive: bool = False
    livekit_archive_dir: Path | None = None
    catalog_topics: frozenset[str] = Field(default_factory=frozenset)
    enabled_capabilities: tuple[str, ...] | None = None
    discovery_source_order: str | None = None
    mastery_threshold: float = 0.75


class DiscoveryArtifact(BaseModel):
    """One provider discovery result."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    kind: str
    paths: tuple[str, ...] = ()
    reputation_score: float = 0.5
    title: str = ""


class VerificationReport(BaseModel):
    """Per-package verification outcome."""

    model_config = ConfigDict(frozen=True)

    package_id: str
    novel_concepts: int = 0
    existing_concepts: int = 0
    conflicting_concepts: int = 0
    adjusted_confidence: float = 0.5
    proposal_hint: str | None = None


class CapabilityMetrics(BaseModel):
    """Measurement for one capability invocation."""

    model_config = ConfigDict(frozen=True)

    capability: str
    level: int
    success: bool = True
    yield_count: int = 0
    duration_ms: float = 0.0
    notes: str = ""
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CapabilityResult(BaseModel):
    """Output of one capability run."""

    model_config = ConfigDict(frozen=True)

    capability: str
    metrics: CapabilityMetrics
    artifacts: dict[str, Any] = Field(default_factory=dict)


class PipelineState(BaseModel):
    """Mutable accumulator across capability stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    discoveries: list[DiscoveryArtifact] = Field(default_factory=list)
    packages: list[KnowledgePackage] = Field(default_factory=list)
    verified_packages: list[KnowledgePackage] = Field(default_factory=list)
    recommendations: list[ResearchRecommendation] = Field(default_factory=list)
    verification_reports: list[VerificationReport] = Field(default_factory=list)
    proposal_hints: list[str] = Field(default_factory=list)
    providers_evaluated: int = 0
    conflicts_detected: int = 0
    curiosity_signals: list = Field(default_factory=list)
    graph_gaps: list = Field(default_factory=list)
    active_missions: list = Field(default_factory=list)
    video_fixtures: list = Field(default_factory=list)
    live_streams: list = Field(default_factory=list)
    multimodal_synced: list = Field(default_factory=list)
    cross_source_report: object | None = None
    curriculum_diagnostics: list = Field(default_factory=list)


class CapabilityContext(BaseModel):
    """Frozen inputs for capability execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    store: RecordStore
    config: ResearcherPipelineConfig
    graph: object | None = None
    state: object | None = None
    identities: dict = Field(default_factory=dict)
    student_ids: tuple[str, ...] = ()
    plan: object | None = None
    kel_findings: tuple = ()
    ecosystem: object | None = None
    strategy_hints: object | None = None


@runtime_checkable
class Capability(Protocol):
    """Composable Researcher skill."""

    level: int
    name: str

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult: ...


def persist_capability_metrics(store: RecordStore, result: CapabilityResult) -> None:
    """Append capability measurement to the versioned store."""
    key = f"{result.capability}::{result.metrics.measured_at.timestamp()}"
    store.put(
        METRICS_NAMESPACE,
        key,
        {
            "capability": result.capability,
            "level": result.metrics.level,
            "success": result.metrics.success,
            "yield_count": result.metrics.yield_count,
            "duration_ms": result.metrics.duration_ms,
            "notes": result.metrics.notes,
        },
        reason=f"capability {result.capability}",
    )
