"""Tests for cross-cue motion continuity."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.motion_continuity import MotionContinuityCapability
from allm.researcher.motion_continuity import continuity_score, link_motion_continuity
from allm.researcher.multimodal import discover_video_fixtures, sync_fixtures_with_workshop_dir
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def _workshop_rows() -> list[SyncedEvidence]:
    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    return [
        row
        for row in sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
        if row.visual is not None
    ]


def test_continuity_score_links_adjacent_magnet_cues() -> None:
    rows = _workshop_rows()
    chase = next(row for row in rows if row.timestamp_sec == 845.0)
    repulsion = next(row for row in rows if row.timestamp_sec == 848.0)
    score = continuity_score(chase, repulsion)
    assert score >= 0.35


def test_link_motion_continuity_groups_workshop_timeline() -> None:
    rows = _workshop_rows()
    enriched, tracks = link_motion_continuity(rows, min_score=0.35)
    multi_track = next((track for track in tracks if len(track.timestamps) >= 2), None)
    assert multi_track is not None
    assert len(multi_track.timestamps) >= 2
    linked = [row for row in enriched if row.linked_cue_timestamps]
    assert linked


def test_motion_continuity_capability_enriches_pipeline() -> None:
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            enable_motion_continuity=True,
            motion_continuity_min_score=0.35,
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = _workshop_rows()
    result = MotionContinuityCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert any(row.motion_track_id for row in pipeline.multimodal_synced)


def test_isolated_cue_gets_track_id_without_links() -> None:
    row = SyncedEvidence(
        source_id="solo-workshop",
        timestamp_sec=10.0,
        transcript_excerpt="unique moment",
        visual=VisualCue(description="Static whiteboard slide", tags=("whiteboard",)),
        confidence=0.7,
    )
    enriched, tracks = link_motion_continuity([row], min_score=0.35)
    assert enriched[0].motion_track_id
    assert enriched[0].linked_cue_timestamps == ()
    assert len(tracks) == 1
