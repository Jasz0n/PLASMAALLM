"""Audit Keshe book PDF corpus readiness (M33)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.book_pdf import extract_pdf_text, pypdf_available, resolve_pdf_backend

CorpusStatus = Literal["readable", "stub", "corrupt", "empty"]

# Expected page ranges for the Keshe trilogy (tolerance for edition front-matter).
EXPECTED_BOOK_PAGES: dict[str, tuple[int, int]] = {
    "Book_1_-_The_Universal_Order_of_Creation_of_Matters.pdf": (210, 220),
    "Book_the_structure_of_Light.pdf": (125, 135),
    "Keshe book3 - The Origin of the Universe.pdf": (140, 150),
}


class BookCorpusEntry(BaseModel):
    """Readiness of one book PDF in the corpus."""

    model_config = ConfigDict(frozen=True)

    filename: str
    size_bytes: int
    status: CorpusStatus
    pages_sampled: int = 0
    text_chars: int = 0
    sidecar_path: str | None = None
    page_count: int | None = None
    pages_ok: bool | None = None
    note: str = ""


class CorpusBootstrapResult(BaseModel):
    """Report from bootstrapping missing book sidecars."""

    model_config = ConfigDict(frozen=True)

    created: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    template_dir: str


def sidecar_path(pdf_path: Path | str) -> Path:
    """Default sidecar text path for one book PDF."""
    return Path(pdf_path).with_suffix(".txt")


def template_path_for_pdf(pdf_path: Path | str, template_dir: Path | str) -> Path:
    """Resolve a template file for one PDF stem."""
    pdf = Path(pdf_path)
    directory = Path(template_dir)
    named = directory / f"{pdf.stem}.txt"
    if named.is_file():
        return named
    return directory / "default_plasma_excerpt.txt"


def bootstrap_book_sidecar(
    pdf_path: Path | str,
    *,
    template_dir: Path | str,
    force: bool = False,
) -> Path | None:
    """Create a ``.txt`` sidecar for stub/corrupt PDFs from templates."""
    pdf = Path(pdf_path)
    if not pdf.is_file():
        return None

    target = sidecar_path(pdf)
    if target.is_file() and not force:
        return None

    entry = audit_book_pdf(pdf)
    if entry.status == "readable" and not force:
        return None

    template = template_path_for_pdf(pdf, template_dir)
    if not template.is_file():
        return None

    target.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def bootstrap_book_corpus(
    books_dir: Path | str,
    *,
    template_dir: Path | str,
    max_files: int | None = None,
    force: bool = False,
) -> CorpusBootstrapResult:
    """Bootstrap sidecars for all non-readable PDFs in a book directory."""
    directory = Path(books_dir)
    templates = Path(template_dir)
    paths = sorted(directory.glob("*.pdf"))
    if max_files is not None:
        paths = paths[:max_files]

    created: list[str] = []
    skipped: list[str] = []
    for pdf_path in paths:
        result = bootstrap_book_sidecar(pdf_path, template_dir=templates, force=force)
        if result is None:
            skipped.append(pdf_path.name)
        else:
            created.append(result.name)
    return CorpusBootstrapResult(
        created=tuple(created),
        skipped=tuple(skipped),
        template_dir=str(templates),
    )


def corpus_ready_count(entries: tuple[BookCorpusEntry, ...]) -> tuple[int, int]:
    """Return (usable, total) where usable includes readable and stub with text."""
    usable = sum(1 for row in entries if row.status in {"readable", "stub"} and row.text_chars > 0)
    return usable, len(entries)


def pdf_page_count(path: Path | str) -> int | None:
    """Return total page count when pypdf can open the PDF."""
    pdf_path = Path(path)
    if not pypdf_available() or not pdf_path.is_file():
        return None
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return None


def pages_within_expected(filename: str, page_count: int | None) -> bool | None:
    """Return whether page count matches expected range, or None if unknown."""
    if page_count is None:
        return None
    bounds = EXPECTED_BOOK_PAGES.get(filename)
    if bounds is None:
        return None
    low, high = bounds
    return low <= page_count <= high


def audit_book_pdf(
    path: Path | str,
    *,
    max_pages: int = 4,
    backend: str = "auto",
) -> BookCorpusEntry:
    """Classify one PDF as readable, stub-backed, corrupt, or empty."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        return BookCorpusEntry(
            filename=pdf_path.name,
            size_bytes=0,
            status="corrupt",
            note="file missing",
        )

    size = pdf_path.stat().st_size
    total_pages = pdf_page_count(pdf_path)
    pages_ok = pages_within_expected(pdf_path.name, total_pages)
    selected = resolve_pdf_backend(backend)
    if selected == "pypdf" and pypdf_available():
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            pages = reader.pages[: max(0, max_pages)]
            chunks: list[str] = []
            for page in pages:
                chunks.append((page.extract_text() or "").strip())
            text = "\n\n".join(chunk for chunk in chunks if chunk)
            if text.strip():
                return BookCorpusEntry(
                    filename=pdf_path.name,
                    size_bytes=size,
                    status="readable",
                    pages_sampled=len(pages),
                    text_chars=len(text),
                    sidecar_path=str(sidecar_path(pdf_path)) if sidecar_path(pdf_path).is_file() else None,
                    page_count=total_pages,
                    pages_ok=pages_ok,
                )
            stub = extract_pdf_text(pdf_path, max_pages=max_pages, backend="stub")
            if stub.strip() and not stub.startswith("[stub pdf]"):
                return BookCorpusEntry(
                    filename=pdf_path.name,
                    size_bytes=size,
                    status="stub",
                    pages_sampled=len(pages),
                    text_chars=len(stub),
                    sidecar_path=str(sidecar_path(pdf_path)),
                    note="pypdf empty; sidecar text used",
                )
            return BookCorpusEntry(
                filename=pdf_path.name,
                size_bytes=size,
                status="empty",
                pages_sampled=len(pages),
                note="no extractable text",
            )
        except Exception as exc:
            stub = extract_pdf_text(pdf_path, max_pages=max_pages, backend="stub")
            if stub.strip() and not stub.startswith("[stub pdf]"):
                return BookCorpusEntry(
                    filename=pdf_path.name,
                    size_bytes=size,
                    status="stub",
                    text_chars=len(stub),
                    sidecar_path=str(sidecar_path(pdf_path)),
                    note=f"pypdf failed ({exc}); sidecar text used",
                )
            if stub.strip() and stub.startswith("[stub pdf]"):
                return BookCorpusEntry(
                    filename=pdf_path.name,
                    size_bytes=size,
                    status="stub",
                    text_chars=len(stub),
                    sidecar_path=str(sidecar_path(pdf_path)) if sidecar_path(pdf_path).is_file() else None,
                    note=f"pypdf failed ({exc}); placeholder stub",
                )
            return BookCorpusEntry(
                filename=pdf_path.name,
                size_bytes=size,
                status="corrupt",
                note=str(exc),
            )

    text = extract_pdf_text(pdf_path, max_pages=max_pages, backend=backend)
    if text.startswith("[stub pdf]"):
        return BookCorpusEntry(
            filename=pdf_path.name,
            size_bytes=size,
            status="stub",
            text_chars=len(text),
            note="stub backend",
        )
    if text.strip():
        return BookCorpusEntry(
            filename=pdf_path.name,
            size_bytes=size,
            status="readable" if selected == "pypdf" else "stub",
            text_chars=len(text),
        )
    return BookCorpusEntry(
        filename=pdf_path.name,
        size_bytes=size,
        status="empty",
        note="no text extracted",
    )


def audit_book_corpus(
    books_dir: Path | str,
    *,
    max_files: int | None = None,
    max_pages: int = 4,
    backend: str = "auto",
) -> tuple[BookCorpusEntry, ...]:
    """Audit all PDFs in a book directory."""
    directory = Path(books_dir)
    paths = sorted(directory.glob("*.pdf"))
    if max_files is not None:
        paths = paths[:max_files]
    return tuple(audit_book_pdf(path, max_pages=max_pages, backend=backend) for path in paths)


def format_corpus_audit(entries: tuple[BookCorpusEntry, ...]) -> str:
    """Human-readable corpus audit table."""
    if not entries:
        return "  (no PDF files found)"
    lines = [
        f"  {'File':<52} {'Pages':>6} {'Status':<10} {'Chars':>6}",
        f"  {'-' * 78}",
    ]
    for entry in entries:
        pages = str(entry.page_count) if entry.page_count is not None else "?"
        if entry.pages_ok is False:
            pages += "!"
        lines.append(
            f"  {entry.filename:<52} {pages:>6} {entry.status:<10} {entry.text_chars:>6}"
        )
        if entry.note:
            lines.append(f"    note: {entry.note}")
    readable = sum(1 for row in entries if row.status == "readable")
    stubbed = sum(1 for row in entries if row.status == "stub" and row.text_chars > 50)
    complete = sum(1 for row in entries if row.pages_ok is True)
    lines.append(f"\n  readable: {readable}/{len(entries)}  sidecar-backed: {stubbed}  page-verified: {complete}")
    return "\n".join(lines)


def corpus_is_complete(entries: tuple[BookCorpusEntry, ...]) -> bool:
    """Return True when all expected books are readable with verified page counts."""
    if len(entries) < len(EXPECTED_BOOK_PAGES):
        return False
    by_name = {row.filename: row for row in entries}
    for name in EXPECTED_BOOK_PAGES:
        row = by_name.get(name)
        if row is None or row.status != "readable" or row.pages_ok is not True:
            return False
    return True
