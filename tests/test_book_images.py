"""Tests for book PDF figure extraction and visual pipeline (M27)."""

from pathlib import Path

from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher.book_evidence import build_book_synced_evidence
from allm.researcher.book_images import (
    BookImageArtifact,
    book_source_id,
    extract_pdf_images,
)
from allm.researcher.capabilities.base import (
    CapabilityContext,
    DiscoveryArtifact,
    PipelineState,
    ResearcherPipelineConfig,
)
from allm.researcher.capabilities.book_images import BookImagesCapability
from allm.researcher.packages import package_from_book_dir
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/books"
REAL_BOOK = ROOT / "books/Book_1_-_The_Universal_Order_of_Creation_of_Matters.pdf"


def test_build_book_synced_evidence_marks_diagrams() -> None:
    artifact = BookImageArtifact(
        book_id="book:plasma_excerpt",
        pdf_name="plasma_excerpt.pdf",
        page_number=2,
        image_name="diagram",
        image_path=str(FIXTURES / "plasma_excerpt_images/diagram.jpg"),
        page_text="Magnetic fields interact through attraction and repulsion.",
    )
    rows = build_book_synced_evidence((artifact,))
    assert len(rows) == 1
    assert rows[0].source_id.startswith("book:")
    assert rows[0].visual is not None
    assert rows[0].visual.is_diagram
    assert rows[0].visual.frame_path


def test_extract_pdf_images_stub_uses_sidecar_dir() -> None:
    artifacts = extract_pdf_images(
        FIXTURES / "plasma_excerpt.pdf",
        FIXTURES / ".cache",
        max_pages=4,
        pdf_backend="stub",
    )
    assert artifacts
    assert Path(artifacts[0].image_path).is_file()


def test_book_images_capability_attaches_to_package() -> None:
    store = SQLiteRecordStore(":memory:")
    package = package_from_book_dir(
        FIXTURES,
        max_files=1,
        max_pages=8,
        pdf_backend="stub",
        curriculum_topic=DEFAULT_TOPIC,
    )
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            book_dir=FIXTURES,
            book_max_files=1,
            book_max_pages=8,
            book_pdf_backend="stub",
            enable_book_images=True,
            book_max_images=4,
            book_images_cache_dir=FIXTURES / ".cache",
            book_curriculum_topic=DEFAULT_TOPIC,
        ),
    )
    pipeline = PipelineState()
    pipeline.packages.append(package)
    pipeline.discoveries.append(
        DiscoveryArtifact(
            provider_id="keshe-books",
            kind="book",
            paths=(str(FIXTURES / "plasma_excerpt.pdf"),),
            reputation_score=0.9,
            title="fixture book",
        )
    )
    result = BookImagesCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.multimodal_synced
    assert pipeline.packages[0].multimodal_evidence


def test_extract_real_book_images_when_available(tmp_path: Path) -> None:
    if not REAL_BOOK.is_file():
        return
    artifacts = extract_pdf_images(
        REAL_BOOK,
        tmp_path,
        max_pages=40,
        max_images=3,
        pdf_backend="auto",
    )
    assert artifacts
    assert book_source_id(REAL_BOOK) == artifacts[0].book_id
    for artifact in artifacts:
        assert Path(artifact.image_path).is_file()
