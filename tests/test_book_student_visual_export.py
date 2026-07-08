"""Tests for book visual briefs through student export (M28)."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.visual_distillation import VisualDistillationCapability
from allm.researcher.capabilities.visual_export import VisualExportCapability
from allm.researcher.multimodal_types import DistilledVisualBrief, SyncedEvidence, VisualCue
from allm.researcher.packages import package_from_book_dir
from allm.researcher.visual_distillation import briefs_for_provider, distill_visual_evidence
from allm.storage import SQLiteRecordStore
from allm.teacher.visual_export import approve_visual_brief, export_approved_briefs

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/books"


def _book_row(page: int, *, hint: str = "magnetic") -> SyncedEvidence:
    return SyncedEvidence(
        source_id="book:plasma_excerpt",
        timestamp_sec=float(page),
        transcript_excerpt=f"Page {page} explains {hint} field interaction in plasma.",
        visual=VisualCue(
            description=f"Book diagram page {page}",
            frame_path=str(FIXTURES / "plasma_excerpt_images/diagram.jpg"),
            is_diagram=True,
            caption=f"Vision: {hint} field diagram on page {page}",
            tags=("book", "diagram"),
        ),
        concept_hints=(hint, "plasma"),
        confidence=0.85,
    )


def _workshop_row() -> SyncedEvidence:
    return SyncedEvidence(
        source_id="workshop9",
        timestamp_sec=712.0,
        transcript_excerpt="Rotating magnets on a pin without fuel.",
        visual=VisualCue(
            description="Workshop magnet demo",
            caption="Vision: magnet rotation demo",
            tags=("demo",),
        ),
        concept_hints=("plasma motion",),
        confidence=0.9,
    )


def test_briefs_for_provider_splits_book_and_workshop() -> None:
    book_briefs = distill_visual_evidence([_book_row(1), _book_row(2)])
    workshop_briefs = distill_visual_evidence([_workshop_row()])
    briefs = book_briefs + workshop_briefs
    assert briefs_for_provider(briefs, "keshe-books")
    assert briefs_for_provider(briefs, "kids-workshops")
    assert all(brief.source_kind == "book" for brief in briefs_for_provider(briefs, "keshe-books"))
    assert all(brief.source_kind == "workshop" for brief in briefs_for_provider(briefs, "kids-workshops"))


def test_book_visual_export_strips_paths() -> None:
    briefs = distill_visual_evidence([_book_row(26, hint="gravitational")])
    assert len(briefs) == 1
    assert briefs[0].source_kind == "book"
    exports = export_approved_briefs(
        briefs,
        (approve_visual_brief(briefs[0], approved_by="teacher-book"),),
        curriculum_topic="kids-plasma",
    )
    assert exports
    payload = exports[0].model_dump()
    assert "frame_path" not in payload
    assert "source_refs" not in payload
    assert all("Book diagram" not in image or "Vision:" in image for image in exports[0].images)


def test_visual_distillation_capability_scopes_book_package() -> None:
    store = SQLiteRecordStore(":memory:")
    package = package_from_book_dir(
        FIXTURES,
        max_files=1,
        max_pages=8,
        pdf_backend="stub",
        curriculum_topic="kids-plasma",
    )
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            enable_visual_distillation=True,
            book_curriculum_topic="kids-plasma",
            workshop_curriculum_topic="kids-plasma",
        ),
    )
    pipeline = PipelineState()
    pipeline.packages = [package]
    pipeline.multimodal_synced = [_book_row(3), _workshop_row()]
    result = VisualDistillationCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 2
    book_briefs = pipeline.packages[0].distilled_visual_briefs
    assert book_briefs
    assert all(brief.source_kind == "book" for brief in book_briefs)


def test_book_visual_export_capability() -> None:
    store = SQLiteRecordStore(":memory:")
    briefs = distill_visual_evidence([_book_row(30, hint="plasmatic")])
    package = package_from_book_dir(
        FIXTURES,
        max_files=1,
        max_pages=8,
        pdf_backend="stub",
        curriculum_topic="kids-plasma",
    )
    from allm.researcher.visual_distillation import attach_distilled_visuals

    package = attach_distilled_visuals(package, briefs)
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            enable_visual_export=True,
            visual_export_auto_approve=True,
            visual_export_min_confidence=0.7,
            book_curriculum_topic="kids-plasma",
        ),
    )
    pipeline = PipelineState()
    pipeline.packages = [package]
    result = VisualExportCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.packages[0].student_visual_packages
