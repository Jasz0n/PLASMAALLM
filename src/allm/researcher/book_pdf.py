"""PDF text extraction for Keshe book discovery (M26)."""

from __future__ import annotations

from pathlib import Path

from allm.core.logging import get_logger

logger = get_logger("researcher.book_pdf")

BACKENDS = ("auto", "pypdf", "stub")


def pypdf_available() -> bool:
    """Return True when pypdf is importable."""
    try:
        import pypdf  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_pdf_backend(requested: str) -> str:
    """Pick a concrete PDF backend from ``auto`` or an explicit name."""
    normalized = requested.lower().strip()
    if normalized not in BACKENDS:
        raise ValueError(f"unknown PDF backend {requested!r}; expected one of {BACKENDS}")
    if normalized == "auto":
        return "pypdf" if pypdf_available() else "stub"
    if normalized == "pypdf" and not pypdf_available():
        logger.warning("pypdf unavailable; falling back to stub PDF backend")
        return "stub"
    return normalized


def extract_pdf_text(
    path: Path | str,
    *,
    max_pages: int = 32,
    backend: str = "auto",
) -> str:
    """Extract plain text from one PDF, capped at ``max_pages``."""
    pdf_path = Path(path)
    selected = resolve_pdf_backend(backend)
    if selected == "pypdf":
        return _extract_with_pypdf(pdf_path, max_pages=max_pages)
    return _extract_with_stub(pdf_path)


def _extract_with_pypdf(path: Path, *, max_pages: int) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        pages = reader.pages[: max(0, max_pages)]
    except Exception as exc:
        logger.warning("pypdf failed for %s: %s; using stub fallback", path.name, exc)
        return _extract_with_stub(path)

    chunks: list[str] = []
    for index, page in enumerate(pages):
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text.strip())
        if index + 1 >= max_pages:
            break
    if chunks:
        return "\n\n".join(chunks)
    logger.warning("pypdf returned no text for %s; using stub fallback", path.name)
    return _extract_with_stub(path)


def _extract_with_stub(path: Path) -> str:
    """Offline stub: read a sidecar ``.txt`` or return a filename placeholder."""
    sidecar = path.with_suffix(".txt")
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8")
    extract_sidecar = path.with_name(f"{path.stem}_extract.txt")
    if extract_sidecar.is_file():
        return extract_sidecar.read_text(encoding="utf-8")
    return f"[stub pdf] {path.stem.replace('_', ' ')}"
