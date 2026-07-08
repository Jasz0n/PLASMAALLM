"""Book-only curriculum splits for KEL tests (no workshop transcripts)."""

from __future__ import annotations

import os
from pathlib import Path

from allm.data.base import Sample
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher.packages import package_from_single_book_pdf
from allm.teacher.source_training import BOOK_PROVIDER, samples_from_book_packages


def books_only_mode() -> bool:
    return os.environ.get("ALLM_BOOKS_ONLY", "0") == "1"


def apply_books_only_defaults() -> None:
    """Configure env for book-only KEL test runs."""
    if not books_only_mode():
        return
    os.environ.setdefault("ALLM_KEL_PHASE_ORDER", "books_only")
    os.environ.setdefault("ALLM_RESEARCHER_WORKSHOP_FILES", "0")
    os.environ.setdefault("ALLM_CROSS_SOURCE_VERIFY", "0")
    os.environ.setdefault("ALLM_WORKSHOP_DELTA", "0")
    os.environ.setdefault("ALLM_MULTIMODAL", "0")


def _book_dir(root: Path) -> Path:
    return Path(os.environ.get("ALLM_BOOK_DIR", str(root / "books")))


def _samples_from_package(
    package,
    *,
    book_name: str,
    topic: str,
) -> list[Sample]:
    rows = samples_from_book_packages((package,), topic=topic)
    tagged: list[Sample] = []
    for row in rows:
        metadata = dict(row.metadata)
        metadata["book"] = book_name
        metadata["sample_kind"] = metadata.get("sample_kind", "definition")
        tagged.append(row.model_copy(update={"metadata": metadata}))
    return tagged


def load_book_curriculum_splits(root: Path) -> tuple[list[Sample], list[Sample]]:
    """Build train/holdout from Keshe books — last book is held out."""
    books_dir = _book_dir(root)
    if not books_dir.is_dir():
        raise FileNotFoundError(books_dir)

    max_files = int(os.environ.get("ALLM_BOOK_MAX_FILES", "3"))
    max_pages = int(os.environ.get("ALLM_BOOK_MAX_PAGES", "64"))
    pdf_backend = os.environ.get("ALLM_BOOK_PDF_BACKEND", "auto")
    topic = os.environ.get("ALLM_BOOK_CURRICULUM_TOPIC", DEFAULT_TOPIC)
    holdout_index = int(os.environ.get("ALLM_BOOK_HOLDOUT_INDEX", "-1"))

    paths = sorted(books_dir.glob("*.pdf"))[:max_files]
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least 2 book PDFs in {books_dir}")

    if holdout_index < 0:
        holdout_index = len(paths) - 1

    train: list[Sample] = []
    holdout: list[Sample] = []
    for index, path in enumerate(paths):
        package = package_from_single_book_pdf(
            path,
            provider=BOOK_PROVIDER,
            curriculum_topic=topic,
            max_pages=max_pages,
            pdf_backend=pdf_backend,
        )
        rows = _samples_from_package(package, book_name=path.name, topic=topic)
        if index == holdout_index:
            holdout.extend(rows)
        else:
            train.extend(rows)

    min_pool = int(os.environ.get("ALLM_BOOK_MIN_SAMPLES", "4"))
    if len(train) < min_pool or len(holdout) < min_pool:
        raise ValueError(
            f"Book pools too small: train={len(train)} holdout={len(holdout)} "
            f"(need >={min_pool} each)"
        )
    return train, holdout


def load_test_curriculum_splits(root: Path) -> tuple[list[Sample], list[Sample]]:
    """Load train/holdout for KEL tests — books only or legacy workshop jsonl."""
    apply_books_only_defaults()
    if books_only_mode():
        return load_book_curriculum_splits(root)
    from allm.kdp.curriculum import load_curriculum_splits

    return load_curriculum_splits(root)
