"""Tests for workshop ↔ book cross-source verification (M29)."""

from pathlib import Path

from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.cross_source import CrossSourceVerificationCapability
from allm.researcher.cross_source import align_workshop_and_book
from allm.researcher.packages import package_from_book_dir, package_from_workshop_dir
from allm.researcher.types import PackageConcept
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/books"


def test_align_workshop_and_book_finds_token_overlap() -> None:
    workshop = package_from_workshop_dir(
        ROOT / "transcripts/Kids/cleaned/mk",
        max_files=2,
        curriculum_topic=DEFAULT_TOPIC,
    )
    book = package_from_book_dir(
        FIXTURES,
        max_files=1,
        max_pages=8,
        pdf_backend="stub",
        curriculum_topic=DEFAULT_TOPIC,
    )
    report = align_workshop_and_book(workshop, book, min_overlap=0.2)
    assert report.workshop_package_id
    assert report.book_package_id
    assert isinstance(report.aligned_count, int)


def test_cross_source_capability_requires_both_providers() -> None:
    store = SQLiteRecordStore(":memory:")
    workshop = package_from_workshop_dir(
        ROOT / "transcripts/Kids/cleaned/mk",
        max_files=1,
        curriculum_topic=DEFAULT_TOPIC,
    )
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            enable_cross_source_verification=True,
            cross_source_min_overlap=0.25,
        ),
    )
    pipeline = PipelineState()
    pipeline.packages = [workshop]
    result = CrossSourceVerificationCapability().run(ctx, pipeline)
    assert result.metrics.yield_count == 0


def test_cross_source_capability_aligns_fixture_packages() -> None:
    store = SQLiteRecordStore(":memory:")
    workshop = package_from_workshop_dir(
        ROOT / "transcripts/Kids/cleaned/mk",
        max_files=1,
        curriculum_topic=DEFAULT_TOPIC,
    )
    book = package_from_book_dir(
        FIXTURES,
        max_files=1,
        max_pages=8,
        pdf_backend="stub",
        curriculum_topic=DEFAULT_TOPIC,
    )
    shared = PackageConcept(name="plasma magnetic fields", description="shared", confidence=0.8)
    book = book.model_copy(
        update={"concepts": book.concepts + (shared,)}
    )
    workshop = workshop.model_copy(
        update={"concepts": workshop.concepts + (shared.model_copy(update={"name": "Plasma Magnetic Fields"}),)}
    )
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            enable_cross_source_verification=True,
            cross_source_min_overlap=0.35,
        ),
    )
    pipeline = PipelineState()
    pipeline.packages = [workshop, book]
    pipeline.verified_packages = [workshop, book]
    result = CrossSourceVerificationCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.cross_source_report is not None
    assert pipeline.cross_source_report.aligned_count >= 1


def test_researcher_cycle_includes_cross_source_capability() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        workshop_max_files=1,
        book_dir=FIXTURES,
        book_max_files=1,
        book_max_pages=8,
        book_pdf_backend="stub",
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()
    names = {row[0] for row in report.capability_summary}
    assert "verification.cross_source" in names
