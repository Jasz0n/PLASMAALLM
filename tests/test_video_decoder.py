"""Tests for auto video fixture generation."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.multimodal import VideoDiscoveryCapability
from allm.researcher.video_decoder import (
    ensure_workshop_fixtures,
    find_video_mentions,
    generate_fixture_from_transcript,
)

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP_DIR = ROOT / "transcripts/Kids/cleaned/mk"
TRANSCRIPT = WORKSHOP_DIR / "knowledgeSeekerWorkshop9.txt"


def test_find_video_mentions_in_workshop9() -> None:
    text = TRANSCRIPT.read_text(encoding="utf-8")
    mentions = find_video_mentions(text)
    assert len(mentions) >= 2


def test_generate_fixture_from_transcript() -> None:
    fixture = generate_fixture_from_transcript(TRANSCRIPT)
    assert fixture is not None
    assert fixture.source_id == "knowledgeSeekerWorkshop9"
    assert len(fixture.cues) >= 2
    assert fixture.cues[0].visual is not None


def test_ensure_workshop_fixtures_writes_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    fixtures = ensure_workshop_fixtures(WORKSHOP_DIR, cache)
    assert fixtures
    assert list(cache.glob("*_auto.json"))


def test_discovery_auto_generates_fixtures(tmp_path: Path) -> None:
    cache = tmp_path / "visual_cache"
    ctx = CapabilityContext(
        store=__import__("allm.storage", fromlist=["SQLiteRecordStore"]).SQLiteRecordStore(":memory:"),
        config=ResearcherPipelineConfig(
            workshop_dir=WORKSHOP_DIR,
            video_fixture_dir=cache,
            auto_generate_video_fixtures=True,
            workshop_max_files=1,
        ),
    )
    cap = VideoDiscoveryCapability()
    result = cap.run(ctx, PipelineState())
    assert result.metrics.yield_count >= 1
    assert cache.glob("*.json")
