"""Tests for LiveKit bridge and live stream observation."""

from __future__ import annotations

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.livekit import LiveKitDiscoveryCapability, LiveStreamObserveCapability
from allm.researcher.livekit_observer import StubLiveKitObserver, get_livekit_observer
from allm.researcher.livekit_tokens import create_livekit_token, load_livekit_config
from allm.researcher.livekit_types import LiveKitConfig
from allm.researcher.social_stream_client import load_livekit_fixture
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "transcripts/Kids/visual/livekit_streams_fixture.json"


def test_create_livekit_token_has_three_segments() -> None:
    config = LiveKitConfig(
        url="wss://example.livekit.cloud",
        api_key="test-key",
        api_secret="test-secret",
    )
    token = create_livekit_token(
        config,
        identity="plasma-researcher",
        room_name="workshop-plasma-live-demo",
    )
    assert token.count(".") == 2


def test_load_livekit_fixture_parses_snapshots() -> None:
    streams = load_livekit_fixture(FIXTURE)
    assert len(streams) == 1
    assert streams[0].snapshots
    assert streams[0].stream_id == "workshop-plasma-live-demo"


def test_livekit_discovery_from_fixture() -> None:
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            enable_livekit=True,
            livekit_fixture_path=FIXTURE,
            livekit_topics=frozenset({"kids-plasma"}),
        ),
    )
    pipeline = PipelineState()
    result = LiveKitDiscoveryCapability().run(ctx, pipeline)
    assert result.metrics.yield_count == 1
    assert pipeline.live_streams[0].title.startswith("Plasma magnet")


def test_livestream_observe_merges_evidence() -> None:
    store = SQLiteRecordStore(":memory:")
    streams = load_livekit_fixture(FIXTURE)
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            enable_livekit=True,
            livekit_observer_backend="stub",
        ),
    )
    pipeline = PipelineState()
    pipeline.live_streams = streams
    result = LiveStreamObserveCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.multimodal_synced
    assert pipeline.multimodal_synced[0].is_live
    assert pipeline.multimodal_synced[0].live_stream_id == "workshop-plasma-live-demo"


def test_stub_observer_backend() -> None:
    observer = get_livekit_observer("stub")
    assert isinstance(observer, StubLiveKitObserver)


def test_load_livekit_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    config = load_livekit_config()
    assert config is not None
    assert config.url.endswith("livekit.cloud")
