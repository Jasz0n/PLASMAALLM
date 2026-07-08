"""Tests for multimodal evidence and synchronization."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.multimodal import MultimodalSyncCapability, VideoDiscoveryCapability
from allm.researcher.multimodal import (
    count_video_mentions,
    discover_video_fixtures,
    load_video_fixture,
    retrieve_synced_evidence,
    sync_fixtures_with_workshop_dir,
    sync_transcript_cues,
)
from allm.researcher.packages import package_from_workshop_dir
from allm.researcher.types import KnowledgePackage
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]
VISUAL_DIR = ROOT / "transcripts/Kids/visual"
WORKSHOP_DIR = ROOT / "transcripts/Kids/cleaned/mk"
FIXTURE = VISUAL_DIR / "workshop9_plasma_demo.json"


def test_count_video_mentions() -> None:
    text = "As you've seen in the video the plasma changed color in this video."
    assert count_video_mentions(text) >= 2


def test_load_video_fixture() -> None:
    fixture = load_video_fixture(FIXTURE)
    assert fixture.source_id == "knowledgeSeekerWorkshop9"
    assert len(fixture.cues) >= 2


def test_sync_transcript_cues_matches_workshop9() -> None:
    fixture = load_video_fixture(FIXTURE)
    transcript = (WORKSHOP_DIR / "knowledgeSeekerWorkshop9.txt").read_text(encoding="utf-8")
    synced = sync_transcript_cues(fixture, transcript)
    assert len(synced) >= 2
    assert synced[0].visual is not None
    assert synced[0].timestamp_sec > 0


def test_sync_fixtures_with_workshop_dir() -> None:
    fixtures = discover_video_fixtures(VISUAL_DIR)
    synced = sync_fixtures_with_workshop_dir(fixtures, WORKSHOP_DIR)
    assert synced
    sources = {row.source_id for row in synced}
    assert "knowledgeSeekerWorkshop9" in sources


def test_retrieve_synced_evidence_by_query() -> None:
    fixtures = discover_video_fixtures(VISUAL_DIR)
    synced = sync_fixtures_with_workshop_dir(fixtures, WORKSHOP_DIR)
    package = KnowledgePackage.build(
        provider="test",
        title="test",
        curriculum_topic="kids-plasma",
        multimodal_evidence=tuple(synced),
    )
    hits = retrieve_synced_evidence(package, query="blue plasma")
    assert hits
    assert hits[0].visual is not None


def test_video_discovery_and_sync_capabilities() -> None:
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=WORKSHOP_DIR,
            video_fixture_dir=VISUAL_DIR,
            workshop_max_files=1,
            workshop_curriculum_topic="kids-plasma",
        ),
    )
    pipeline = PipelineState()
    package = package_from_workshop_dir(
        WORKSHOP_DIR,
        max_files=5,
        curriculum_topic="kids-plasma",
    )
    pipeline.packages.append(package)

    video_result = VideoDiscoveryCapability().run(ctx, pipeline)
    assert video_result.metrics.yield_count == 2
    assert pipeline.video_fixtures

    sync_result = MultimodalSyncCapability().run(ctx, pipeline)
    assert sync_result.metrics.yield_count >= 5
    assert pipeline.packages[0].multimodal_evidence
