"""Tests for Keshe book corpus audit (M33)."""

from pathlib import Path

from allm.researcher.book_corpus import (
    audit_book_corpus,
    audit_book_pdf,
    bootstrap_book_corpus,
    bootstrap_book_sidecar,
    format_corpus_audit,
)

ROOT = Path(__file__).resolve().parents[1]


def test_audit_book_pdf_stub_fixture() -> None:
    fixture = ROOT / "tests/fixtures/books/plasma_excerpt.pdf"
    if not fixture.is_file():
        sidecar = fixture.with_suffix(".txt")
        if not sidecar.is_file():
            return
    entry = audit_book_pdf(fixture, backend="stub")
    assert entry.filename.endswith(".pdf")
    assert entry.status in {"readable", "stub", "empty"}


def test_audit_book_corpus_lists_entries(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% not a real pdf\n")
    sidecar = pdf.with_suffix(".txt")
    sidecar.write_text("Plasma field diagram excerpt for testing.", encoding="utf-8")

    entries = audit_book_corpus(tmp_path, max_files=1, backend="stub")
    assert len(entries) == 1
    assert entries[0].status in {"stub", "corrupt", "readable"}
    assert entries[0].text_chars > 0
    report = format_corpus_audit(entries)
    assert "sample.pdf" in report


def test_corpus_is_complete_with_expected_pages() -> None:
    from allm.researcher.book_corpus import EXPECTED_BOOK_PAGES, BookCorpusEntry, corpus_is_complete

    entries = []
    for name, (low, high) in EXPECTED_BOOK_PAGES.items():
        entries.append(
            BookCorpusEntry(
                filename=name,
                size_bytes=1_000_000,
                status="readable",
                text_chars=500,
                page_count=(low + high) // 2,
                pages_ok=True,
            )
        )
    assert corpus_is_complete(tuple(entries))


def test_bootstrap_book_sidecar_from_template(tmp_path: Path) -> None:
    pdf = tmp_path / "Book_the_structure_of_Light.pdf"
    pdf.write_bytes(b"%PDF-1.4\nbroken")
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "Book_the_structure_of_Light.txt").write_text(
        "Light structure plasma fields for workshop alignment.",
        encoding="utf-8",
    )

    created = bootstrap_book_sidecar(pdf, template_dir=templates)
    assert created is not None
    assert created.is_file()
    entry = audit_book_pdf(pdf)
    assert entry.status == "stub"
    assert entry.text_chars > 20

    result = bootstrap_book_corpus(tmp_path, template_dir=templates)
    assert result.created == ()
    assert pdf.name in result.skipped
