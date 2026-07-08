"""Tests for Keshe book PDF discovery (M26)."""

from pathlib import Path

from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.book_pdf import extract_pdf_text, resolve_pdf_backend
from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.discovery import BookDiscoveryCapability
from allm.researcher.packages import package_from_book_dir
from allm.researcher.providers import BookProvider
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/books"


def test_book_provider_discovers_pdfs() -> None:
    provider = BookProvider(FIXTURES)
    paths = provider.discover()
    assert len(paths) == 1
    assert paths[0].name == "plasma_excerpt.pdf"


def test_extract_pdf_text_stub_reads_sidecar() -> None:
    text = extract_pdf_text(
        FIXTURES / "plasma_excerpt.pdf",
        backend="stub",
    )
    assert "plasma field" in text.lower()
    assert "magnetic" in text.lower()


def test_package_from_book_dir_builds_concepts() -> None:
    package = package_from_book_dir(
        FIXTURES,
        max_files=1,
        max_pages=8,
        pdf_backend="stub",
        curriculum_topic=DEFAULT_TOPIC,
    )
    assert package.provider == "keshe-books"
    assert package.curriculum_topic == DEFAULT_TOPIC
    assert package.concepts


def test_book_discovery_capability() -> None:
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            book_dir=FIXTURES,
            book_max_files=1,
        ),
    )
    pipeline = PipelineState()
    result = BookDiscoveryCapability().run(ctx, pipeline)
    assert result.metrics.yield_count == 1
    assert pipeline.discoveries[0].kind == "book"


def test_researcher_cycle_with_books() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        workshop_max_files=0,
        book_dir=FIXTURES,
        book_max_files=1,
        book_max_pages=8,
        book_pdf_backend="stub",
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()
    book_packages = [pkg for pkg in report.packages if pkg.provider == "keshe-books"]
    assert book_packages
    assert report.recommendations
    names = {row[0] for row in report.capability_summary}
    assert "discovery.book" in names


def test_resolve_pdf_backend_auto_or_stub() -> None:
    backend = resolve_pdf_backend("auto")
    assert backend in {"pypdf", "stub"}
