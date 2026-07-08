"""Tests for vision caption enrichment."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.vision import VisionEnrichmentCapability
from allm.researcher.multimodal import sync_fixtures_with_workshop_dir
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.researcher.vision_caption import StubVisionCaptioner, enrich_synced_evidence
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_stub_captioner_produces_caption() -> None:
    captioner = StubVisionCaptioner()
    text = captioner.caption(
        transcript_excerpt="as you've seen in the video the plasma twists",
        tags=("blue-plasma",),
        concept_hints=("magnetical beat",),
    )
    assert "Transcript-aligned" in text
    assert "plasma" in text.lower()


def test_enrich_synced_evidence_adds_caption() -> None:
    row = SyncedEvidence(
        source_id="knowledgeSeekerWorkshop9",
        timestamp_sec=845.0,
        transcript_excerpt="as you've seen in the video the one magnet",
        visual=VisualCue(
            description="Blue field between magnets",
            frame_start=2145,
            frame_end=2189,
            tags=("blue-plasma",),
        ),
        confidence=0.87,
    )
    enriched = enrich_synced_evidence(row, captioner=StubVisionCaptioner())
    assert enriched.visual is not None
    assert enriched.visual.caption
    assert enriched.confidence >= row.confidence


def test_vision_capability_enriches_pipeline() -> None:
    from allm.researcher.multimodal import discover_video_fixtures

    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    synced = sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            enable_vision_captions=True,
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = synced
    result = VisionEnrichmentCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.multimodal_synced[0].visual.caption
