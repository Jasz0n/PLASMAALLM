"""M41 capstone — dependency risk + retrieval strength + book review budget.

    PYTHONPATH=src python3 examples/69_m41_planner_retrieval_kel.py
    PYTHONPATH=src ALLM_STUDENT_SIZE=large python3 examples/69_m41_planner_retrieval_kel.py
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
from allm.researcher.book_corpus import audit_book_corpus, corpus_is_complete, format_corpus_audit
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
        "ALLM_TEACHER_UI_APPROVAL": "1",
        "ALLM_VISUAL_DELIVERY": "1",
        "ALLM_BOOTSTRAP": "1",
        "ALLM_KEL_PHASE_ORDER": "books_then_workshops",
        "ALLM_KEL_BOOK_ITERS": "4",
        "ALLM_KEL_WORKSHOP_ITERS": "4",
        "ALLM_MAINTENANCE_CURRICULUM": "1",
        "ALLM_ADAPTIVE_MAINTENANCE": "1",
        "ALLM_KS_PLANNER": "1",
        "ALLM_DEPENDENCY_RISK": "1",
        "ALLM_KS_PROGRESSION": "1",
        "ALLM_KS_REPAIR_HALT": "1",
        "ALLM_WORKSHOP_DELTA": "1",
        "ALLM_FORGETTING_WATCHDOG": "1",
        "ALLM_LOOP_SEED": "42",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="M41 planner retrieval capstone")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ALLM_LOOP_SEED", "42")))
    args = parser.parse_args()

    books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
    entries = audit_book_corpus(books_dir, max_files=3, max_pages=8)
    print("\n=== M41: dependency risk + retrieval strength + book review budget ===")
    print(format_corpus_audit(entries))
    if not corpus_is_complete(entries):
        raise SystemExit("Book corpus incomplete.")

    workdir = Path(tempfile.mkdtemp(prefix="allm-m41-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")
    os.environ["ALLM_LOOP_SEED"] = str(args.seed)

    result = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=workdir,
        verbose=True,
    )

    print("\n=== M41 summary ===")
    print(f"  iterations completed:  {result.iterations_completed}")
    print(f"  exam score:            {result.first_score:.2f} -> {result.last_score:.2f}")
    print(f"  held-out gain:         {result.heldout_gain:+.2f}")
    print(f"  peak score:            {result.peak_score:.2f}")
    print(f"  KEL LG:                {result.kel_lg}")
    print(f"  KEL KS:                {result.kel_ks}")
    print(f"  artifacts:             {result.workdir}")


if __name__ == "__main__":
    main()
