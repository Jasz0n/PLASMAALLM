"""Tests for M17 LiveKit production integration."""

from __future__ import annotations

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.livekit import LiveKitArchiveCapability, LiveStreamObserveCapability
from allm.researcher.livekit_archive import archive_evidence_rows, evidence_to_fixture
from allm.researcher.livekit_tokens import create_livekit_token, load_livekit_config
from allm.researcher.livekit_types import LiveKitConfig
from allm.researcher.livekit_worker import LiveKitObserverWorker, reset_livekit_worker
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.researcher.social_stream_client import load_livekit_fixture
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "transcripts/Kids/visual/livekit_streams_fixture.json"


def test_archive_evidence_writes_fixture(tmp_path: Path) -> None:
    rows = [
        SyncedEvidence(
            source_id="livekit:demo",
            timestamp_sec=5.0,
            transcript_excerpt="magnet rotation live",
            visual=VisualCue(description="Live magnets", tags=("livekit",)),
            live_stream_id="demo",
            is_live=True,
            confidence=0.85,
        )
    ]
    path = archive_evidence_rows(
        stream_id="demo",
        title="Demo stream",
        curriculum_topic="kids-plasma",
        evidence=rows,
        output_dir=tmp_path,
    )
    assert path is not None
    assert path.is_file()
    assert "livekit_demo" in path.name


def test_evidence_to_fixture_builds_cues() -> None:
    rows = [
        SyncedEvidence(
            source_id="livekit:demo",
            timestamp_sec=1.0,
            transcript_excerpt="a",
            is_live=True,
            live_stream_id="demo",
        ),
        SyncedEvidence(
            source_id="livekit:demo",
            timestamp_sec=9.0,
            transcript_excerpt="b",
            is_live=True,
            live_stream_id="demo",
        ),
    ]
    fixture = evidence_to_fixture(stream_id="demo", title="Demo", evidence=rows)
    assert len(fixture.cues) == 2
    assert fixture.duration_sec >= 60.0


def test_livekit_worker_buffers_and_archives(tmp_path: Path) -> None:
    reset_livekit_worker()
    worker = LiveKitObserverWorker(tmp_path)
    row = SyncedEvidence(
        source_id="livekit:demo",
        timestamp_sec=2.0,
        transcript_excerpt="buffered",
        is_live=True,
        live_stream_id="demo",
    )
    worker.append("demo", [row], title="Demo", curriculum_topic="kids-plasma")
    assert len(worker.buffered("demo")) == 1
    archived = worker.archive_stream("demo", archive_dir=tmp_path / "archives")
    assert archived is not None
    assert worker.buffered("demo") == []


def test_archive_capability_archives_live_rows(tmp_path: Path) -> None:
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            enable_livekit=True,
            enable_livekit_archive=True,
            livekit_archive_dir=tmp_path,
            workshop_curriculum_topic="kids-plasma",
        ),
    )
    pipeline = PipelineState()
    pipeline.live_streams = load_livekit_fixture(FIXTURE)
    pipeline.multimodal_synced = [
        SyncedEvidence(
            source_id="livekit:workshop-plasma-live-demo",
            timestamp_sec=12.0,
            transcript_excerpt="live moment",
            is_live=True,
            live_stream_id="workshop-plasma-live-demo",
        )
    ]
    result = LiveKitArchiveCapability().run(ctx, pipeline)
    assert result.metrics.yield_count == 1
    assert list(tmp_path.glob("*_archive.json"))


def test_observe_uses_stub_with_fixture() -> None:
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


def test_create_livekit_token_roundtrip() -> None:
    config = LiveKitConfig(url="wss://x.livekit.cloud", api_key="k", api_secret="s")
    token = create_livekit_token(config, identity="plasma-researcher", room_name="room-1")
    assert token.count(".") == 2
