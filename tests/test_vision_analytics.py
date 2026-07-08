"""Tests for vision analytics enrichment."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.vision_analytics import VisionAnalyticsCapability
from allm.researcher.multimodal import discover_video_fixtures, sync_fixtures_with_workshop_dir
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.researcher.vision_analytics import (
    StubVisionAnalyzer,
    enrich_synced_evidence_analytics,
    get_vision_analyzer,
)
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_stub_analyzer_detects_motion_and_color() -> None:
    analyzer = StubVisionAnalyzer()
    result = analyzer.analyze(
        description="One magnet chasing the other — blue field region visible between poles",
        tags=("blue-plasma", "magnet-chase", "field-beat"),
        transcript_excerpt="as you've seen in the video the one magnet",
        frame_start=2145,
        frame_end=2189,
    )
    assert result.motion_level in {"high", "moderate"}
    assert "blue" in result.dominant_colors
    assert "motion_high" in result.visual_features or "motion_moderate" in result.visual_features


def test_stub_analyzer_detects_diagram_from_labels() -> None:
    analyzer = StubVisionAnalyzer()
    result = analyzer.analyze(
        description="Close-up of repulsion between similar poles",
        tags=("repulsion", "similar-poles"),
        diagram_labels=("field lines", "poles"),
    )
    assert result.is_diagram is True
    assert "diagram" in result.visual_features


def test_enrich_synced_evidence_adds_analytics() -> None:
    row = SyncedEvidence(
        source_id="knowledgeSeekerWorkshop9",
        timestamp_sec=845.0,
        transcript_excerpt="as you've seen in the video the one magnet",
        visual=VisualCue(
            description="One magnet chasing the other — blue field region visible between poles",
            frame_start=2145,
            frame_end=2189,
            tags=("blue-plasma", "magnet-chase"),
        ),
        confidence=0.87,
    )
    enriched = enrich_synced_evidence_analytics(row, analyzer=StubVisionAnalyzer())
    assert enriched.visual is not None
    assert enriched.visual.motion_level
    assert enriched.visual.dominant_colors
    assert enriched.visual.analytics_summary
    assert enriched.confidence > row.confidence


def test_vision_analytics_capability_enriches_pipeline() -> None:
    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    synced = sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            enable_vision_analytics=True,
            vision_analytics_backend="stub",
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = [row for row in synced if row.visual is not None]
    result = VisionAnalyticsCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.multimodal_synced[0].visual.visual_features


def test_get_vision_analyzer_stub_backend() -> None:
    analyzer = get_vision_analyzer("stub")
    assert isinstance(analyzer, StubVisionAnalyzer)
