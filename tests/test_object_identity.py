"""Tests for cross-workshop object identity persistence."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.object_identity import ObjectIdentityCapability
from allm.researcher.motion_continuity import enrich_synced_evidence_continuity
from allm.researcher.motion_tracking import StubMotionTracker, enrich_synced_evidence_motion
from allm.researcher.multimodal import discover_video_fixtures, sync_fixtures_with_workshop_dir
from allm.researcher.object_identity import cross_source_identity_score, link_object_identities
from allm.researcher.object_identity import _build_track_aggregates
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def _continuity_rows() -> list:
    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    synced = sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
    visual_rows = [row for row in synced if row.visual is not None]
    motion_rows = [
        enrich_synced_evidence_motion(row, tracker=StubMotionTracker())
        for row in visual_rows
    ]
    enriched, _tracks = enrich_synced_evidence_continuity(motion_rows, min_score=0.35)
    return enriched


def test_workshop3_fixture_syncs() -> None:
    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    workshop3 = next(fixture for fixture in fixtures if fixture.source_id == "knowledgeSeekerWorkshop3")
    synced = sync_fixtures_with_workshop_dir([workshop3], ROOT / "transcripts/Kids/cleaned/mk")
    assert len(synced) >= 2


def test_cross_source_identity_links_magnet_workshops() -> None:
    rows = _continuity_rows()
    enriched, identities = link_object_identities(rows, min_score=0.30)
    multi_source = next((record for record in identities if len(record.source_ids) >= 2), None)
    assert multi_source is not None
    assert "knowledgeSeekerWorkshop3" in multi_source.source_ids
    assert "knowledgeSeekerWorkshop9" in multi_source.source_ids
    linked = [row for row in enriched if row.linked_source_ids]
    assert linked


def test_cross_source_identity_score_positive_for_magnet_tracks() -> None:
    rows = _continuity_rows()
    aggregates = _build_track_aggregates(rows)
    workshop3 = next(item for item in aggregates if item.source_id == "knowledgeSeekerWorkshop3")
    workshop9 = next(item for item in aggregates if item.source_id == "knowledgeSeekerWorkshop9")
    score = cross_source_identity_score(workshop3, workshop9)
    assert score >= 0.30


def test_object_identity_capability_enriches_pipeline() -> None:
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            enable_object_identity=True,
            enable_motion_continuity=True,
            object_identity_min_score=0.30,
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = _continuity_rows()
    result = ObjectIdentityCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert any(row.object_identity_id for row in pipeline.multimodal_synced)
