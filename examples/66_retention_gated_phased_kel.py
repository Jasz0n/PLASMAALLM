"""Retention-gated phased KEL capstone (M38).

Builds on M37 phased books→workshops learning with:
- held-out retention gates before KEL strategy advance
- forgetting watchdog on topic mastery
- pinned book definition notes (survive workshop context crowding)
- workshop delta samples (workshop-only concepts vs book graph)

    PYTHONPATH=src python3 examples/66_retention_gated_phased_kel.py
    PYTHONPATH=src python3 examples/66_retention_gated_phased_kel.py --compare
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
from kids_kel_run import KidsKelRunResult, run_kids_kel_steered


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
        "ALLM_KEL_RETENTION_GATE": "1",
        "ALLM_KEL_RETENTION_MAX_DROP": "0.15",
        "ALLM_FORGETTING_WATCHDOG": "1",
        "ALLM_WORKSHOP_DELTA": "1",
        "ALLM_LOOP_SEED": "42",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _verify_corpus() -> None:
    books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
    entries = audit_book_corpus(books_dir, max_files=3, max_pages=8)
    print("\n=== M38: Keshe trilogy corpus verification ===")
    print(format_corpus_audit(entries))
    if not corpus_is_complete(entries):
        raise SystemExit("Book corpus incomplete — need all three Keshe PDFs in books/.")


def _run_arm(phase_order: str, *, seed: int) -> KidsKelRunResult:
    os.environ["ALLM_KEL_PHASE_ORDER"] = phase_order
    os.environ["ALLM_LOOP_SEED"] = str(seed)
    os.environ.pop("ALLM_DISCOVERY_ORDER", None)
    workdir = Path(tempfile.mkdtemp(prefix=f"allm-m38-{phase_order}-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")
    print(f"\n{'=' * 60}")
    print(f"=== M38 retention-gated arm: {phase_order} (seed={seed}) ===")
    print(f"{'=' * 60}")
    return run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=workdir,
        verbose=True,
    )


def _print_summary(label: str, result: KidsKelRunResult) -> None:
    print(f"\n--- {label} ---")
    print(f"  phase order:           {result.learning_phase_order}")
    print(f"  cross-source aligned:  {result.aligned_concepts}")
    print(f"  visual notes delivered:{result.visual_notes_delivered}")
    print(f"  iterations completed:  {result.iterations_completed}")
    print(f"  exam score:            {result.first_score:.2f} -> {result.last_score:.2f}")
    print(f"  held-out gain:         {result.heldout_gain:+.2f}")
    print(f"  peak score:            {result.peak_score:.2f}")
    print(f"  KEL LG:                {result.kel_lg}")
    print(f"  artifacts:             {result.workdir}")


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="Retention-gated phased KEL (M38)")
    parser.add_argument("--compare", action="store_true", help="compare both phase orders")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ALLM_LOOP_SEED", "42")))
    args = parser.parse_args()

    _verify_corpus()

    if args.compare:
        books_first = _run_arm("books_then_workshops", seed=args.seed)
        workshops_first = _run_arm("workshops_then_books", seed=args.seed)
        _print_summary("books → workshops (M38)", books_first)
        _print_summary("workshops → books (M38)", workshops_first)
        delta = books_first.heldout_gain - workshops_first.heldout_gain
        print("\n=== M38 comparison ===")
        print(f"  books-first gain:      {books_first.heldout_gain:+.2f}")
        print(f"  workshops-first gain:  {workshops_first.heldout_gain:+.2f}")
        print(f"  delta:                 {delta:+.2f}")
        print(f"  better order:          {'books → workshops' if delta > 0 else 'workshops → books' if delta < 0 else 'tie'}")
    else:
        result = _run_arm(os.environ["ALLM_KEL_PHASE_ORDER"], seed=args.seed)
        _print_summary(os.environ["ALLM_KEL_PHASE_ORDER"], result)


if __name__ == "__main__":
    main()
