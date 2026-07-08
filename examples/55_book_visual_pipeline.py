"""Book PDF figures + vision enrichment demo (M27).

Extracts embedded diagrams from Keshe books, captions them, and
distills Teacher-ready visual briefs — same path as workshop video.

    PYTHONPATH=src python3 examples/55_book_visual_pipeline.py

Environment:
    ALLM_BOOK_MAX_FILES=1
    ALLM_BOOK_MAX_PAGES=40
    ALLM_BOOK_MAX_IMAGES=12
    ALLM_VISION_CAPTIONS=1
    ALLM_FRAME_OCR=1
    ALLM_VISUAL_DISTILL=1
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

    workdir = Path(tempfile.mkdtemp(prefix="allm-book-visual-"))
    store = SQLiteRecordStore(workdir / "book-visual.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_max_files=0,
        book_dir=books_dir,
        book_max_files=int(os.environ.get("ALLM_BOOK_MAX_FILES", "1")),
        book_max_pages=int(os.environ.get("ALLM_BOOK_MAX_PAGES", "40")),
        book_max_images=int(os.environ.get("ALLM_BOOK_MAX_IMAGES", "12")),
        book_pdf_backend=os.environ.get("ALLM_BOOK_PDF_BACKEND", "auto"),
        enable_book_images=True,
        book_images_cache_dir=workdir / "book_images",
        enable_vision_captions=os.environ.get("ALLM_VISION_CAPTIONS", "1") == "1",
        enable_frame_ocr=os.environ.get("ALLM_FRAME_OCR", "1") == "1",
        enable_visual_distillation=os.environ.get("ALLM_VISUAL_DISTILL", "1") == "1",
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()

    print("\n=== M27: Book figures + visual pipeline ===")
    book_packages = [pkg for pkg in report.packages if pkg.provider == "keshe-books"]
    print(f"  book packages: {len(book_packages)}")
    print(f"  multimodal synced: {len(report.multimodal_synced)}")

    book_figures = [row for row in report.multimodal_synced if row.source_id.startswith("book:")]
    print(f"  book figures: {len(book_figures)}")
    for row in book_figures[:5]:
        visual = row.visual
        caption = visual.caption if visual else None
        path_note = "cached image" if visual and visual.frame_path else "no image"
        print(f"\n    {row.source_id} page {int(row.timestamp_sec)}")
        print(f"      {path_note}")
        if caption:
            print(f"      caption: {caption[:100]}...")
        if visual and visual.ocr_text:
            print(f"      ocr: {visual.ocr_text[:80]}...")

    brief_count = sum(len(pkg.distilled_visual_briefs) for pkg in book_packages)
    print(f"\n  distilled visual briefs: {brief_count}")
    for package in book_packages:
        for brief in package.distilled_visual_briefs[:2]:
            print(f"    brief {brief.brief_id}: {brief.concept_name} ({len(brief.images)} images)")

    for name, yield_count, notes in report.capability_summary:
        if "book" in name or name in {
            "understanding.vision",
            "understanding.ocr",
            "understanding.visual.distill",
        }:
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Cached figures under {workdir / 'book_images'}")


if __name__ == "__main__":
    main()
