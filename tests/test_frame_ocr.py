"""Tests for frame OCR enrichment."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.ocr import OcrEnrichmentCapability
from allm.researcher.frame_ocr import StubFrameOcr, enrich_synced_evidence_ocr, get_frame_ocr
from allm.researcher.multimodal import sync_fixtures_with_workshop_dir
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_stub_ocr_infers_diagram_labels() -> None:
    ocr = StubFrameOcr()
    text, labels = ocr.read_frame(
        frame_path=None,
        description="One magnet chasing the other — blue field region visible between poles",
        tags=("blue-plasma", "magnet-chase", "field-beat"),
        concept_hints=("magnetical beat",),
        transcript_excerpt="as you've seen in the video the one magnet",
    )
    assert labels
    assert "blue plasma" in labels or "blue-plasma" in " ".join(labels)
    assert text


def test_enrich_synced_evidence_adds_ocr() -> None:
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
    enriched = enrich_synced_evidence_ocr(row, ocr=StubFrameOcr())
    assert enriched.visual is not None
    assert enriched.visual.diagram_labels
    assert enriched.visual.ocr_text
    assert enriched.confidence > row.confidence


def test_ocr_capability_enriches_pipeline() -> None:
    from allm.researcher.multimodal import discover_video_fixtures

    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    synced = sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            enable_frame_ocr=True,
            ocr_backend="stub",
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = [row for row in synced if row.visual is not None]
    result = OcrEnrichmentCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.multimodal_synced[0].visual.diagram_labels


def test_get_frame_ocr_stub_backend() -> None:
    ocr = get_frame_ocr("stub")
    assert isinstance(ocr, StubFrameOcr)
