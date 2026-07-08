"""Tests for books-only KEL test curriculum."""

from __future__ import annotations

from pathlib import Path

import pytest

from allm.data.base import Sample
from allm.kdp.book_curriculum import apply_books_only_defaults, books_only_mode
from allm.researcher.types import KnowledgePackage


def test_books_only_defaults(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_BOOKS_ONLY", "1")
    apply_books_only_defaults()
    assert books_only_mode()
    import os

    assert os.environ["ALLM_KEL_PHASE_ORDER"] == "books_only"
    assert os.environ["ALLM_RESEARCHER_WORKSHOP_FILES"] == "0"


def test_load_book_curriculum_splits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALLM_BOOKS_ONLY", "1")
    monkeypatch.setenv("ALLM_BOOK_DIR", str(tmp_path))
    monkeypatch.setenv("ALLM_BOOK_MAX_FILES", "3")
    monkeypatch.setenv("ALLM_BOOK_MIN_SAMPLES", "1")

    for index in range(3):
        (tmp_path / f"book{index}.pdf").write_bytes(b"%PDF-1.4")

    packages = [
        KnowledgePackage(
            id=f"book{index}",
            provider="keshe-books",
            title=f"book{index}.pdf",
            curriculum_topic="kids-plasma",
            definitions=((f"term{index}", f"def{index}"),),
        )
        for index in range(3)
    ]

    def fake_single(path, **kwargs):
        name = Path(path).name
        index = int(name.replace("book", "").replace(".pdf", ""))
        return packages[index]

    monkeypatch.setattr(
        "allm.kdp.book_curriculum.package_from_single_book_pdf",
        fake_single,
    )

    from allm.kdp.book_curriculum import load_book_curriculum_splits

    train, holdout = load_book_curriculum_splits(tmp_path.parent)
    assert holdout
    assert train
    assert all(row.metadata.get("source") == "book" for row in (*train, *holdout))
    assert holdout[0].metadata.get("book") == "book2.pdf"
