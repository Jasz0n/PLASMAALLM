"""Keshe book PDF discovery demo (M26).

Discovers PDFs under ``books/``, extracts text, runs KDP distillation,
and enqueues curriculum recommendations for Teacher review.

    PYTHONPATH=src python3 examples/54_book_discovery.py

Environment:
    ALLM_BOOK_MAX_FILES=1     — limit PDFs per cycle (default 1)
    ALLM_BOOK_MAX_PAGES=32    — pages extracted per PDF
    ALLM_BOOK_PDF_BACKEND=auto — pypdf when installed, else stub sidecar
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    books_dir = ROOT / "books"
    if not books_dir.is_dir():
        raise SystemExit(f"Book directory not found: {books_dir}")

    workdir = Path(tempfile.mkdtemp(prefix="allm-book-discovery-"))
    store = SQLiteRecordStore(workdir / "books.sqlite3")

    max_files = int(os.environ.get("ALLM_BOOK_MAX_FILES", "1"))
    max_pages = int(os.environ.get("ALLM_BOOK_MAX_PAGES", "32"))
    backend = os.environ.get("ALLM_BOOK_PDF_BACKEND", "auto")

    researcher = ResearcherLayer(
        store,
        workshop_max_files=0,
        book_dir=books_dir,
        book_max_files=max_files,
        book_max_pages=max_pages,
        book_pdf_backend=backend,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()

    print("\n=== M26: Keshe book PDF discovery ===")
    book_packages = [pkg for pkg in report.packages if pkg.provider == "keshe-books"]
    print(f"  book packages: {len(book_packages)}")
    for package in book_packages:
        print(f"\n  package: {package.id}")
        print(f"    title: {package.title}")
        print(f"    concepts: {len(package.concepts)}")
        print(f"    conflicts: {len(package.conflicts)}")
        print(f"    confidence: {package.confidence:.2f}")
        for concept in package.concepts[:5]:
            print(f"      - {concept.name[:72]}")

    for name, yield_count, notes in report.capability_summary:
        if name in {"discovery.book", "understanding.package", "curriculum.target"}:
            print(f"  capability: {name} yield={yield_count} ({notes})")

    topics = {rec.topic for rec in report.recommendations if rec.provider == "keshe-books"}
    print(f"\n  book recommendation topics: {len(topics)}")
    for topic in sorted(topics)[:8]:
        print(f"    - {topic}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
