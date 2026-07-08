"""Tests for audio feature enrichment."""

from pathlib import Path

from allm.researcher.audio_analysis import (
    StubAudioAnalyzer,
    enrich_synced_evidence_audio,
    get_audio_analyzer,
)
from allm.researcher.capabilities.audio import AudioEnrichmentCapability
from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.multimodal import sync_fixtures_with_workshop_dir
from allm.researcher.multimodal_types import AudioCue, SyncedEvidence
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_stub_analyzer_maps_machine_tags() -> None:
    analyzer = StubAudioAnalyzer()
    features = analyzer.analyze(
        description="Soft mechanical click as magnets twist",
        tags=("machine-sound", "magnet-click"),
        transcript_excerpt="we've just seen in this video",
    )
    assert "machine_sound" in features.features
    assert features.analysis


def test_enrich_synced_evidence_adds_audio_features() -> None:
    row = SyncedEvidence(
        source_id="knowledgeSeekerWorkshop9",
        timestamp_sec=712.0,
        transcript_excerpt="we've just seen in this video",
        audio=AudioCue(
            description="Soft mechanical click as magnets twist",
            tags=("machine-sound", "magnet-click"),
        ),
        confidence=0.87,
    )
    enriched = enrich_synced_evidence_audio(row, analyzer=StubAudioAnalyzer())
    assert enriched.audio is not None
    assert enriched.audio.features
    assert enriched.audio.analysis
    assert enriched.confidence > row.confidence


def test_audio_capability_enriches_pipeline() -> None:
    from allm.researcher.multimodal import discover_video_fixtures

    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    synced = sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            enable_audio_analysis=True,
            audio_analysis_backend="stub",
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = [row for row in synced if row.audio is not None]
    result = AudioEnrichmentCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.multimodal_synced[0].audio.features


def test_get_audio_analyzer_stub_backend() -> None:
    analyzer = get_audio_analyzer("stub")
    assert isinstance(analyzer, StubAudioAnalyzer)
