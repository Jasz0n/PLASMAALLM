"""M46 — Researcher model router + curriculum diagnostics on KEL remediation.

    PYTHONPATH=src python3 examples/74_m46_curriculum_diagnostics_kel.py
    PYTHONPATH=src ALLM_STUDENT_SIZE=large python3 examples/74_m46_curriculum_diagnostics_kel.py
"""

from __future__ import annotations

import argparse
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
from allm.researcher.model_router import resolve_model_spec
from allm.researcher.ollama_vision import ollama_reachable
from kids_kel_run import run_kids_kel_steered


def _apply_defaults() -> None:
    defaults = {
        "ALLM_RESEARCHER": "1",
        "ALLM_MULTIMODAL": "0",
        "ALLM_BOOK_DISCOVERY": "1",
        "ALLM_BOOK_IMAGES": "1",
        "ALLM_BOOK_BOOTSTRAP_SIDECARS": "auto",
        "ALLM_VISION_CAPTIONS": "1",
        "ALLM_VISION_BACKEND": "auto",
        "ALLM_VISION_MODEL": "llava",
        "ALLM_FRAME_OCR": "1",
        "ALLM_OCR_BACKEND": "auto",
        "ALLM_OCR_MODEL": "llava",
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
        "ALLM_KEL_RESEARCH_REQUESTS": "1",
        "ALLM_RESEARCHER_MODEL_ROUTER": "1",
        "ALLM_CURRICULUM_DIAGNOSTICS": "1",
        "ALLM_CURRICULUM_DIAGNOSTICS_BACKEND": "auto",
        "ALLM_RESEARCHER_REASONING_MODEL": "qwen2.5:14b-instruct",
        "ALLM_RESEARCHER_VERIFIER_MODEL": "qwen2.5:14b-instruct",
        "ALLM_KEL_STRATEGY_DIVERSITY": "1",
        "ALLM_KS_PROGRESSION": "1",
        "ALLM_KS_REPAIR_HALT": "1",
        "ALLM_WORKSHOP_DELTA": "0",
        "ALLM_FORGETTING_WATCHDOG": "1",
        "ALLM_LOOP_SEED": "42",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _print_diagnostics(workdir: Path) -> None:
    path = workdir / "kids.sqlite3"
    if not path.is_file():
        return
    from allm.storage import SQLiteRecordStore

    store = SQLiteRecordStore(path)
    keys = store.keys("curriculum_diagnostics")
    if not keys:
        print("\n=== Curriculum diagnostics ===")
        print("  (none persisted — enable ALLM_CURRICULUM_DIAGNOSTICS=1)")
        store.close()
        return
    print("\n=== Curriculum diagnostics (Chief Scientist) ===")
    for key in sorted(keys)[-6:]:
        record = store.get("curriculum_diagnostics", key)
        if record is None:
            continue
        row = record.value
        print(
            f"  [{row['trigger']}] {row['failure_reason']} "
            f"({row['confidence']:.2f}) via {row['specialist_role']}/{row['model_id']}"
        )
        if row.get("evidence"):
            print(f"    evidence: {row['evidence'][0]}")
        if row.get("recommendations"):
            print(f"    fix: {row['recommendations'][0]}")
    store.close()


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="M46 curriculum diagnostics capstone")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ALLM_LOOP_SEED", "42")))
    args = parser.parse_args()

    books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
    entries = audit_book_corpus(books_dir, max_files=3, max_pages=8)
    reasoning = resolve_model_spec("reasoning")
    verifier = resolve_model_spec("verifier")
    vision_model = os.environ.get("ALLM_VISION_MODEL", "llava")
    print("\n=== M46: Researcher model router + curriculum diagnostics ===")
    print(f"  reasoning: {reasoning.model_id}")
    print(f"  verifier:  {verifier.model_id}")
    print(f"  vision:    {vision_model} (ollama reachable: {ollama_reachable()})")
    print(format_corpus_audit(entries))
    if not corpus_is_complete(entries):
        raise SystemExit("Book corpus incomplete.")

    workdir = Path(tempfile.mkdtemp(prefix="allm-m46-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")
    os.environ["ALLM_LOOP_SEED"] = str(args.seed)

    result = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=workdir,
        verbose=True,
    )

    _print_diagnostics(workdir)

    final_input = EvaluationInput(
        student_id="kids-kel",
        kel_ks=result.kel_ks,
        kel_lg=result.kel_lg,
        heldout_first=result.first_score,
        heldout_last=result.last_score,
        heldout_peak=result.peak_score,
    )
    bench = BenchmarkSuite().evaluate(final_input)
    compromise = compromise_decision(IndependentEvaluator().evaluate(final_input))

    print("\n=== Benchmark suite ===")
    print(BenchmarkSuite.summarize(bench))
    print("\n=== Multi-objective compromise ===")
    print(f"  mode:   {compromise.mode}")
    print(f"  score:  {compromise.compromise_score:.2f}")

    print("\n=== M46 summary ===")
    print(f"  iterations completed:  {result.iterations_completed}")
    print(f"  exam score:            {result.first_score:.2f} -> {result.last_score:.2f}")
    print(f"  held-out gain:         {result.heldout_gain:+.2f}")
    print(f"  peak score:            {result.peak_score:.2f}")
    print(f"  KEL KS:                {result.kel_ks}")
    print(f"  compromise mode:       {compromise.mode}")
    print(f"  artifacts:             {result.workdir}")
    print("\n  Check logs for: diagnostics.curriculum and KEL submitted N research request(s)")


if __name__ == "__main__":
    main()
