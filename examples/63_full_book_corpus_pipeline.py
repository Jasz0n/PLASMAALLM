"""Full 3-book corpus pipeline: audit, bootstrap, Researcher, optional KEL (M35).

    PYTHONPATH=src python3 examples/63_full_book_corpus_pipeline.py
    PYTHONPATH=src python3 examples/63_full_book_corpus_pipeline.py --kel
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
    bootstrap_book_corpus,
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
        "ALLM_VISUAL_DISTILL": "1",
        "ALLM_CROSS_SOURCE_VERIFY": "1",
        "ALLM_RESEARCHER_WORKSHOP_FILES": "2",
        "ALLM_BOOK_MAX_FILES": "3",
        "ALLM_BOOK_MAX_PAGES": "24",
        "ALLM_BOOK_MAX_IMAGES": "6",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def run_researcher_only(workdir: Path) -> None:
    """Run Researcher cycle after corpus bootstrap (no KEL loop)."""
    books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
    template_dir = Path(
        os.environ.get("ALLM_BOOK_SIDECAR_TEMPLATES", str(ROOT / "books" / "sidecar_templates"))
    )

    print("\n=== M35 Phase 1: corpus audit (before bootstrap) ===")
    before = audit_book_corpus(books_dir, max_files=int(os.environ.get("ALLM_BOOK_MAX_FILES", "3")))
    print(format_corpus_audit(before))

    print("\n=== M35 Phase 2: bootstrap sidecars ===")
    bootstrap = bootstrap_book_corpus(
        books_dir,
        template_dir=template_dir,
        max_files=int(os.environ.get("ALLM_BOOK_MAX_FILES", "3")),
    )
    print(f"  created: {bootstrap.created or '(none)'}")
    print(f"  skipped: {len(bootstrap.skipped)} pdf(s)")

    print("\n=== M35 Phase 3: corpus audit (after bootstrap) ===")
    after = audit_book_corpus(books_dir, max_files=int(os.environ.get("ALLM_BOOK_MAX_FILES", "3")))
    print(format_corpus_audit(after))
    usable, total = corpus_ready_count(after)
    print(f"  corpus usable for KDP: {usable}/{total}")

    store = SQLiteRecordStore(workdir / "corpus.sqlite3")
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        book_dir=books_dir,
        workshop_max_files=int(os.environ.get("ALLM_RESEARCHER_WORKSHOP_FILES", "2")),
        book_max_files=int(os.environ.get("ALLM_BOOK_MAX_FILES", "3")),
        book_max_pages=int(os.environ.get("ALLM_BOOK_MAX_PAGES", "24")),
        book_max_images=int(os.environ.get("ALLM_BOOK_MAX_IMAGES", "6")),
        enable_book_images=True,
        book_images_cache_dir=workdir / "book_images",
        enable_vision_captions=True,
        enable_visual_distillation=True,
        enable_visual_export=False,
        catalog_topics=(DEFAULT_TOPIC,),
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        frames_cache_dir=workdir / "frames",
    )

    print("\n=== M35 Phase 4: Researcher full corpus cycle ===")
    report = researcher.run_cycle()
    book_pkgs = sum(1 for pkg in report.packages if pkg.provider == "keshe-books")
    workshop_pkgs = sum(1 for pkg in report.packages if pkg.provider == "kids-workshops")
    briefs = sum(len(pkg.distilled_visual_briefs) for pkg in report.packages)
    aligned = getattr(report.cross_source_report, "aligned_count", 0) if report.cross_source_report else 0

    print(f"  packages: {len(report.packages)} (workshop={workshop_pkgs}, book={book_pkgs})")
    print(f"  distilled briefs: {briefs}")
    print(f"  cross-source aligned: {aligned}")
    for package in report.packages:
        if package.provider == "keshe-books":
            print(f"    book package {package.id}: {len(package.concepts)} concepts")
    store.close()


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="Full 3-book corpus pipeline (M35)")
    parser.add_argument(
        "--kel",
        action="store_true",
        help="continue into Teacher-approved KEL capstone after corpus run",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="allm-full-corpus-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")

    if args.kel:
        os.environ.setdefault("ALLM_TEACHER_UI_APPROVAL", "1")
        os.environ.setdefault("ALLM_VISUAL_DELIVERY", "1")
        os.environ.setdefault("ALLM_ITERATIONS", "2")
        os.environ.setdefault("ALLM_BOOTSTRAP", "1")
        print("\n=== M35: full corpus + KEL capstone ===")
        result = run_kids_kel_steered(
            identity_path="configs/students/kids_kel_plasma.yaml",
            workdir=workdir,
            verbose=True,
        )
        print("\n=== KEL summary ===")
        print(f"  book packages:         {result.book_packages}")
        print(f"  cross-source aligned:  {result.aligned_concepts}")
        print(f"  student visual exports:{result.student_visual_exports}")
        print(f"  exam score:            {result.first_score:.2f} -> {result.last_score:.2f}")
    else:
        run_researcher_only(workdir)

    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
