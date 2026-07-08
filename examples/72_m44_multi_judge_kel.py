"""M44 capstone — multi-dimensional grading + curriculum-bound judge.

    PYTHONPATH=src python3 examples/72_m44_multi_judge_kel.py
    PYTHONPATH=src ALLM_STUDENT_SIZE=large python3 examples/72_m44_multi_judge_kel.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.benchmarks.suite import BenchmarkSuite
from allm.core.logging import setup_logging
from allm.evaluator.independent import IndependentEvaluator
from allm.evaluator.types import EvaluationInput
from allm.kel.objectives import compromise_decision
from allm.researcher.book_corpus import audit_book_corpus, corpus_is_complete, format_corpus_audit
from kids_kel_run import run_kids_kel_steered


def _apply_defaults() -> None:
    defaults = {
        "ALLM_RESEARCHER": "1",
        "ALLM_MULTIMODAL": "0",
        "ALLM_BOOK_DISCOVERY": "1",
        "ALLM_BOOK_IMAGES": "1",
        "ALLM_BOOK_BOOTSTRAP_SIDECARS": "auto",
        "ALLM_VISION_CAPTIONS": "1",
        "ALLM_FRAME_OCR": "1",
        "ALLM_VISUAL_DISTILL": "1",
        "ALLM_CROSS_SOURCE_VERIFY": "0",
        "ALLM_RESEARCHER_WORKSHOP_FILES": "0",
        "ALLM_BOOK_MAX_FILES": "3",
        "ALLM_BOOK_MAX_PAGES": "64",
        "ALLM_BOOK_MAX_IMAGES": "36",
        "ALLM_TEACHER_UI_APPROVAL": "1",
        "ALLM_VISUAL_DELIVERY": "1",
        "ALLM_BOOTSTRAP": "1",
        "ALLM_BOOKS_ONLY": "1",
        "ALLM_KEL_PHASE_ORDER": "books_only",
        "ALLM_KEL_BOOK_ITERS": "8",
        "ALLM_MAINTENANCE_CURRICULUM": "1",
        "ALLM_ADAPTIVE_MAINTENANCE": "1",
        "ALLM_KS_PLANNER": "1",
        "ALLM_DEPENDENCY_RISK": "1",
        "ALLM_DECAY_PREDICTION": "1",
        "ALLM_MAINTENANCE_OPTIMIZER": "1",
        "ALLM_MULTI_OBJECTIVE_KEL": "1",
        "ALLM_MULTI_JUDGE": "1",
        "ALLM_KS_PROGRESSION": "1",
        "ALLM_KS_REPAIR_HALT": "1",
        "ALLM_WORKSHOP_DELTA": "0",
        "ALLM_FORGETTING_WATCHDOG": "1",
        "ALLM_LOOP_SEED": "42",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _count_disagreements(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="M44 multi-judge capstone")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ALLM_LOOP_SEED", "42")))
    args = parser.parse_args()

    books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
    entries = audit_book_corpus(books_dir, max_files=3, max_pages=8)
    print("\n=== M44: multi-dimensional grading + curriculum-bound judge ===")
    print(format_corpus_audit(entries))
    if not corpus_is_complete(entries):
        raise SystemExit("Book corpus incomplete.")

    workdir = Path(tempfile.mkdtemp(prefix="allm-m44-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")
    os.environ["ALLM_LOOP_SEED"] = str(args.seed)

    result = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=workdir,
        verbose=True,
    )

    disagreement_path = result.workdir / "judge_disagreements.jsonl"
    disagreement_count = _count_disagreements(disagreement_path)

    final_input = EvaluationInput(
        student_id="kids-kel",
        kel_ks=result.kel_ks,
        kel_lg=result.kel_lg,
        heldout_first=result.first_score,
        heldout_last=result.last_score,
        heldout_peak=result.peak_score,
    )
    bench = BenchmarkSuite().evaluate(final_input)
    snapshot = IndependentEvaluator().evaluate(final_input)
    compromise = compromise_decision(snapshot)

    print("\n=== Benchmark suite (multi-dimensional) ===")
    print(BenchmarkSuite.summarize(bench))
    print(f"\n=== Judge disagreement review dataset ===")
    print(f"  records:   {disagreement_count}")
    print(f"  path:      {disagreement_path}")
    if disagreement_count:
        sample = json.loads(disagreement_path.read_text(encoding="utf-8").splitlines()[0])
        print(f"  sample:    curriculum={sample.get('curriculum_score')} "
              f"alignment={sample.get('alignment')} evidence={sample.get('evidence_score')}")

    print(f"\n=== Multi-objective compromise ===")
    print(f"  mode:   {compromise.mode}")
    print(f"  score:  {compromise.compromise_score:.2f}")

    print("\n=== M44 summary ===")
    print(f"  iterations completed:  {result.iterations_completed}")
    print(f"  exam score:            {result.first_score:.2f} -> {result.last_score:.2f}")
    print(f"  held-out gain:         {result.heldout_gain:+.2f}")
    print(f"  peak score:            {result.peak_score:.2f}")
    print(f"  KEL KS:                {result.kel_ks}")
    print(f"  judge disagreements:   {disagreement_count}")
    print(f"  artifacts:             {result.workdir}")


if __name__ == "__main__":
    main()
