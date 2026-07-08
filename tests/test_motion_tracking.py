"""Tests for temporal motion tracking."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.motion_tracking import MotionTrackingCapability
from allm.researcher.frame_extractor import sample_timestamps_for_span
from allm.researcher.motion_tracking import (
    StubMotionTracker,
    enrich_synced_evidence_motion,
    get_motion_tracker,
)
from allm.researcher.multimodal import discover_video_fixtures, sync_fixtures_with_workshop_dir
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_sample_timestamps_for_span() -> None:
    timestamps = sample_timestamps_for_span(
        timestamp_sec=845.0,
        frame_start=2145,
        frame_end=2189,
        sample_count=3,
        fps=30.0,
    )
    assert len(timestamps) == 3
    assert timestamps[0] == 845.0
    assert timestamps[-1] > timestamps[0]


def test_stub_tracker_detects_rotation() -> None:
    tracker = StubMotionTracker()
    result = tracker.track(
        description="Rotating magnets on a pin — no fuel, motion from field interaction",
        tags=("magnet-rotation", "plasma-motion", "no-fuel"),
        motion_level="high",
        frame_start=1980,
        frame_end=2055,
    )
    assert result.motion_vector == "rotation"
    assert result.motion_score >= 0.5


def test_stub_tracker_detects_oscillation() -> None:
    tracker = StubMotionTracker()
    result = tracker.track(
        description="One magnet chasing the other — blue field region visible between poles",
        tags=("blue-plasma", "magnet-chase", "field-beat"),
        motion_level="high",
        frame_start=2145,
        frame_end=2189,
    )
    assert result.motion_vector == "oscillation"
    assert result.motion_score >= 0.5


def test_enrich_synced_evidence_adds_motion() -> None:
    row = SyncedEvidence(
        source_id="knowledgeSeekerWorkshop9",
        timestamp_sec=845.0,
        transcript_excerpt="as you've seen in the video the one magnet",
        visual=VisualCue(
            description="One magnet chasing the other — blue field region visible between poles",
            frame_start=2145,
            frame_end=2189,
            tags=("blue-plasma", "magnet-chase"),
            motion_level="high",
        ),
        confidence=0.87,
    )
    enriched = enrich_synced_evidence_motion(row, tracker=StubMotionTracker())
    assert enriched.visual is not None
    assert enriched.visual.motion_vector
    assert enriched.visual.motion_score is not None
    assert enriched.visual.motion_summary
    assert enriched.confidence > row.confidence


def test_motion_tracking_capability_enriches_pipeline() -> None:
    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    synced = sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            enable_motion_tracking=True,
            motion_tracking_backend="stub",
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = [row for row in synced if row.visual is not None]
    result = MotionTrackingCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.multimodal_synced[0].visual.motion_vector


def test_get_motion_tracker_stub_backend() -> None:
    tracker = get_motion_tracker("stub")
    assert isinstance(tracker, StubMotionTracker)
