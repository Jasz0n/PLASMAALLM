"""Verified full Keshe trilogy corpus + Researcher + KEL capstone (M36).

Expects complete PDFs in ``books/`` (Book 1 ~216pp, Book 2 ~131pp, Book 3 ~145pp).
Skips sidecar bootstrap when all three pass page verification.

    # Researcher + verification only (faster)
    PYTHONPATH=src python3 examples/64_verified_full_corpus.py

    # Full capstone with Teacher approval + KEL loop
    PYTHONPATH=src python3 examples/64_verified_full_corpus.py --kel
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.book_corpus import (
    audit_book_corpus,
    corpus_is_complete,
    corpus_ready_count,
    format_corpus_audit,
)
from allm.storage import SQLiteRecordStore
from kids_kel_run import run_kids_kel_steered


def _apply_defaults() -> None:
    defaults = {
        "ALLM_RESEARCHER": "1",
        "ALLM_MULTIMODAL": "1",
        "ALLM_BOOK_DISCOVERY": "1",
        "ALLM_BOOK_IMAGES": "1",
        "ALLM_BOOK_BOOTSTRAP_SIDECARS": "auto",
        "ALLM_VISION_CAPTIONS": "1",
        "ALLM_FRAME_OCR": "1",
        "ALLM_VISUAL_DISTILL": "1",
        "ALLM_CROSS_SOURCE_VERIFY": "1",
        "ALLM_RESEARCHER_WORKSHOP_FILES": "3",
        "ALLM_BOOK_MAX_FILES": "3",
        "ALLM_BOOK_MAX_PAGES": "64",
        "ALLM_BOOK_MAX_IMAGES": "36",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def verify_corpus() -> bool:
    """Print audit and return whether the trilogy is complete."""
    books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
    entries = audit_book_corpus(books_dir, max_files=3, max_pages=8)
    print("\n=== M36: Keshe trilogy corpus verification ===")
    print(format_corpus_audit(entries))
    usable, total = corpus_ready_count(entries)
    print(f"  corpus usable: {usable}/{total}")
    complete = corpus_is_complete(entries)
    print(f"  trilogy complete: {complete}")
    for entry in entries:
        if entry.pages_ok is False:
            print(f"  WARNING: {entry.filename} page count {entry.page_count} outside expected range")
    return complete


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="Verified full Keshe corpus capstone (M36)")
    parser.add_argument("--kel", action="store_true", help="run KEL loop after Researcher")
    args = parser.parse_args()

    if not verify_corpus():
        raise SystemExit(
            "Book corpus incomplete. Ensure all three PDFs are in PLASMAALLM/books/ "
            "with full page counts (Book 1 ~216, Book 2 ~131, Book 3 ~145)."
        )

    workdir = Path(tempfile.mkdtemp(prefix="allm-verified-corpus-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")

    if args.kel:
        os.environ.setdefault("ALLM_TEACHER_UI_APPROVAL", "1")
        os.environ.setdefault("ALLM_VISUAL_DELIVERY", "1")
        os.environ.setdefault("ALLM_ITERATIONS", "2")
        os.environ.setdefault("ALLM_BOOTSTRAP", "1")
        print("\n=== M36: verified corpus + Teacher-approved KEL capstone ===")
        result = run_kids_kel_steered(
            identity_path="configs/students/kids_kel_plasma.yaml",
            workdir=workdir,
            verbose=True,
        )
        print("\n=== Capstone summary ===")
        print(f"  book packages:         {result.book_packages}")
        print(f"  cross-source aligned:  {result.aligned_concepts}")
        print(f"  book figures:          {result.book_figures}")
        print(f"  student visual exports:{result.student_visual_exports}")
        print(f"  visual notes delivered:  {result.visual_notes_delivered}")
        print(f"  exam score:            {result.first_score:.2f} -> {result.last_score:.2f}")
        print(f"\nArtifacts: {result.workdir}")
    else:
        print("\n=== M36: verified corpus Researcher cycle ===")
        books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
        store = SQLiteRecordStore(workdir / "verified.sqlite3")
        researcher = ResearcherLayer(
            store,
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            book_dir=books_dir,
            workshop_max_files=int(os.environ.get("ALLM_RESEARCHER_WORKSHOP_FILES", "3")),
            book_max_files=3,
            book_max_pages=int(os.environ.get("ALLM_BOOK_MAX_PAGES", "64")),
            book_max_images=int(os.environ.get("ALLM_BOOK_MAX_IMAGES", "36")),
            enable_book_images=True,
            book_images_cache_dir=workdir / "book_images",
            enable_vision_captions=True,
            enable_visual_distillation=True,
            enable_visual_export=False,
            catalog_topics=(DEFAULT_TOPIC,),
            video_fixture_dir=ROOT / "transcripts/Kids/visual",
            frames_cache_dir=workdir / "frames",
        )
        report = researcher.run_cycle()
        book_pkgs = sum(1 for pkg in report.packages if pkg.provider == "keshe-books")
        workshop_pkgs = sum(1 for pkg in report.packages if pkg.provider == "kids-workshops")
        briefs = sum(len(pkg.distilled_visual_briefs) for pkg in report.packages)
        figures = sum(
            1 for row in report.multimodal_synced if str(row.source_id).startswith("book:")
        )
        aligned = getattr(report.cross_source_report, "aligned_count", 0) if report.cross_source_report else 0
        print("\n=== Researcher summary ===")
        print(f"  packages: {len(report.packages)} (workshop={workshop_pkgs}, book={book_pkgs})")
        print(f"  book figures:          {figures}")
        print(f"  distilled briefs:      {briefs}")
        print(f"  cross-source aligned:  {aligned}")
        store.close()
        print(f"\nArtifacts: {workdir}")


if __name__ == "__main__":
    main()
