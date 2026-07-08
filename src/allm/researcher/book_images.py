"""Extract diagrams and figures from Keshe book PDFs (M27)."""

from __future__ import annotations

import io
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.researcher.book_pdf import extract_pdf_text, pypdf_available, resolve_pdf_backend

logger = get_logger("researcher.book_images")

_TOKEN = re.compile(r"[a-z0-9]+")
_MAX_EDGE = 1280


class BookImageArtifact(BaseModel):
    """One extracted figure from a book PDF page."""

    model_config = ConfigDict(frozen=True)

    book_id: str
    pdf_name: str
    page_number: int = Field(ge=1)
    image_name: str
    image_path: str
    page_text: str = ""


def pillow_available() -> bool:
    """Return True when Pillow is importable."""
    try:
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


def book_source_id(pdf_path: Path | str) -> str:
    """Stable source id for one book PDF."""
    return f"book:{Path(pdf_path).stem}"


def book_images_cache_dir(
    books_dir: Path | str,
    *,
    override: Path | str | None = None,
) -> Path:
    """Default cache location for extracted book figures."""
    if override is not None:
        return Path(override)
    return Path(books_dir) / ".images_cache"


def _page_texts(path: Path, *, max_pages: int, backend: str) -> dict[int, str]:
    if resolve_pdf_backend(backend) != "pypdf" or not pypdf_available():
        full = extract_pdf_text(path, max_pages=max_pages, backend=backend)
        return {1: full} if full.strip() else {}

    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        texts: dict[int, str] = {}
        for index, page in enumerate(reader.pages[: max(0, max_pages)], start=1):
            text = (page.extract_text() or "").strip()
            if text:
                texts[index] = text
        return texts
    except Exception as exc:
        logger.warning("pypdf page text failed for %s: %s; using stub fallback", path.name, exc)
        full = extract_pdf_text(path, max_pages=max_pages, backend="stub")
        return {1: full} if full.strip() else {}


def _save_pil_image(image, destination: Path) -> None:
    from PIL import Image

    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > _MAX_EDGE:
        scale = _MAX_EDGE / longest
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=85, optimize=True)


def extract_pdf_images(
    path: Path | str,
    cache_dir: Path | str,
    *,
    max_pages: int = 32,
    max_images: int = 24,
    pdf_backend: str = "auto",
) -> tuple[BookImageArtifact, ...]:
    """Extract embedded figures from one PDF into JPEG cache files."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        return ()

    backend = resolve_pdf_backend(pdf_backend)
    if backend != "pypdf" or not pypdf_available() or not pillow_available():
        return _extract_stub_images(pdf_path, cache_dir)

    from pypdf import PdfReader
    from PIL import Image

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        logger.warning("pypdf image extract failed for %s: %s; using stub fallback", pdf_path.name, exc)
        return _extract_stub_images(pdf_path, cache_dir)

    page_texts = _page_texts(pdf_path, max_pages=max_pages, backend="pypdf")
    book_id = book_source_id(pdf_path)
    out_dir = Path(cache_dir) / pdf_path.stem
    artifacts: list[BookImageArtifact] = []

    try:
        pages = reader.pages[: max(0, max_pages)]
    except Exception as exc:
        logger.warning("pypdf pages failed for %s: %s; using stub fallback", pdf_path.name, exc)
        return _extract_stub_images(pdf_path, cache_dir)

    for page_index, page in enumerate(pages, start=1):
        if len(artifacts) >= max_images:
            break
        for image_name, image_file in page.images.items():
            if len(artifacts) >= max_images:
                break
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(image_name))
            destination = out_dir / f"page_{page_index:03d}_{safe_name}.jpg"
            if not destination.is_file():
                try:
                    pil = Image.open(io.BytesIO(image_file.data))
                    _save_pil_image(pil, destination)
                except (OSError, ValueError) as exc:
                    logger.warning("book image extract failed %s p%d: %s", pdf_path.name, page_index, exc)
                    continue
            artifacts.append(
                BookImageArtifact(
                    book_id=book_id,
                    pdf_name=pdf_path.name,
                    page_number=page_index,
                    image_name=str(image_name),
                    image_path=str(destination.resolve()),
                    page_text=page_texts.get(page_index, "")[:500],
                )
            )

    logger.info(
        "extracted %d book image(s) from %s (pages<=%d)",
        len(artifacts),
        pdf_path.name,
        max_pages,
    )
    return tuple(artifacts)


def _extract_stub_images(pdf_path: Path, cache_dir: Path | str) -> tuple[BookImageArtifact, ...]:
    """Offline stub: reuse cached JPEGs or sidecar PNG/JPG fixtures."""
    out_dir = Path(cache_dir) / pdf_path.stem
    sidecar_dir = pdf_path.parent / f"{pdf_path.stem}_images"
    book_id = book_source_id(pdf_path)
    page_text = extract_pdf_text(pdf_path, max_pages=8, backend="stub")[:500]
    artifacts: list[BookImageArtifact] = []

    sources: list[Path] = []
    if out_dir.is_dir():
        sources.extend(sorted(out_dir.glob("*.jpg")))
    if sidecar_dir.is_dir():
        sources.extend(sorted(sidecar_dir.glob("*.jpg")))
        sources.extend(sorted(sidecar_dir.glob("*.png")))
    sidecar = pdf_path.with_suffix(".txt")
    if sidecar.is_file():
        page_text = sidecar.read_text(encoding="utf-8")[:500]

    for index, source in enumerate(sources[:8], start=1):
        artifacts.append(
            BookImageArtifact(
                book_id=book_id,
                pdf_name=pdf_path.name,
                page_number=index,
                image_name=source.stem,
                image_path=str(source.resolve()),
                page_text=page_text,
            )
        )
    return tuple(artifacts)
