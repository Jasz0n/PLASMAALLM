"""Phased KEL learning: books first, then workshops (M37).

Compares held-out exam gains when students learn from the verified Keshe trilogy
before kids workshops versus the reverse order.

    # Books then workshops (default)
    PYTHONPATH=src python3 examples/65_phased_books_workshops_kel.py

    # Compare both orders with the same loop seed
    PYTHONPATH=src python3 examples/65_phased_books_workshops_kel.py --compare
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
        "ALLM_KEL_BOOK_ITERS": "2",
        "ALLM_KEL_WORKSHOP_ITERS": "2",
        "ALLM_LOOP_SEED": "42",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _verify_corpus() -> None:
    books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
    entries = audit_book_corpus(books_dir, max_files=3, max_pages=8)
    print("\n=== M37: Keshe trilogy corpus verification ===")
    print(format_corpus_audit(entries))
    if not corpus_is_complete(entries):
        raise SystemExit(
            "Book corpus incomplete. Place all three Keshe PDFs in PLASMAALLM/books/."
        )


def _run_arm(phase_order: str, *, seed: int, verbose: bool) -> KidsKelRunResult:
    os.environ["ALLM_KEL_PHASE_ORDER"] = phase_order
    os.environ["ALLM_LOOP_SEED"] = str(seed)
    os.environ.pop("ALLM_DISCOVERY_ORDER", None)
    workdir = Path(tempfile.mkdtemp(prefix=f"allm-phased-{phase_order}-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")
    print(f"\n{'=' * 60}")
    print(f"=== Phased KEL arm: {phase_order} (seed={seed}) ===")
    print(f"{'=' * 60}")
    return run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=workdir,
        verbose=verbose,
    )


def _print_summary(label: str, result: KidsKelRunResult) -> None:
    print(f"\n--- {label} ---")
    print(f"  phase order:           {result.learning_phase_order}")
    print(f"  book packages:         {result.book_packages}")
    print(f"  workshop packages:     {result.workshop_packages}")
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
    parser = argparse.ArgumentParser(description="Phased books/workshops KEL experiment (M37)")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run books_then_workshops and workshops_then_books with the same seed",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ALLM_LOOP_SEED", "42")))
    args = parser.parse_args()

    _verify_corpus()
    os.environ.setdefault("ALLM_KEL_PHASE_ORDER", "books_then_workshops")

    if args.compare:
        books_first = _run_arm("books_then_workshops", seed=args.seed, verbose=True)
        workshops_first = _run_arm("workshops_then_books", seed=args.seed, verbose=True)
        _print_summary("books → workshops", books_first)
        _print_summary("workshops → books", workshops_first)
        delta = books_first.heldout_gain - workshops_first.heldout_gain
        winner = "books → workshops" if delta > 0 else "workshops → books"
        if delta == 0:
            winner = "tie"
        print("\n=== M37 comparison ===")
        print(f"  books-first gain:      {books_first.heldout_gain:+.2f}")
        print(f"  workshops-first gain:  {workshops_first.heldout_gain:+.2f}")
        print(f"  delta (books - workshop): {delta:+.2f}")
        print(f"  better order:          {winner}")
    else:
        result = _run_arm(
            os.environ["ALLM_KEL_PHASE_ORDER"],
            seed=args.seed,
            verbose=True,
        )
        _print_summary(os.environ["ALLM_KEL_PHASE_ORDER"], result)


if __name__ == "__main__":
    main()
