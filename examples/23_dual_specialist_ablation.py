"""Dual-specialist vs generalist on mixed plasma + software corpus.

Compares three arms on domain-specific held-out exams:
  - generalist: one student, merged pool (no mission filter)
  - plasma-student: plasma train/holdout only
  - software-student: software train/holdout only

    # Offline (ScriptedStudent, seconds)
    PYTHONPATH=src python3 examples/23_dual_specialist_ablation.py --dry-run

    # LLM (Ollama, slow)
    ALLM_ITERATIONS=3 PYTHONPATH=src python3 examples/23_dual_specialist_ablation.py
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

from allm.core.logging import setup_logging
from allm.kdp.mixed_corpus import load_mixed_corpus
from allm.students import load_identity
from dual_specialist_run import SpecialistRunResult, run_specialist_loop

STUDENTS = ROOT / "configs/students"


def main() -> None:
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Dual specialist vs generalist ablation")
    parser.add_argument("--dry-run", action="store_true", help="offline ScriptedStudent")
    args = parser.parse_args()
    dry_run = args.dry_run or os.environ.get("ALLM_DRY_RUN", "0") == "1"

    corpus = load_mixed_corpus(ROOT)
    plasma_id = load_identity(STUDENTS / "plasma_student.yaml")
    software_id = load_identity(STUDENTS / "software_student.yaml")
    base = Path(tempfile.mkdtemp(prefix="allm-dual-spec-"))

    print("\n=== Mixed corpus ===")
    print(f"  plasma:   train={len(corpus.plasma.train)} holdout={len(corpus.plasma.holdout)}")
    print(f"  software: train={len(corpus.software.train)} holdout={len(corpus.software.holdout)}")
    print(f"  merged:   train={len(corpus.merged_train)} holdout={len(corpus.merged_holdout)}")

    generalist = run_specialist_loop(
        student_id="generalist",
        domain="mixed",
        train=corpus.merged_train,
        holdout=corpus.plasma.holdout,
        identity=None,
        dry_run=dry_run,
        workdir=base / "generalist-plasma-exam",
        verbose=True,
    )
    generalist_sw = run_specialist_loop(
        student_id="generalist-sw",
        domain="mixed",
        train=corpus.merged_train,
        holdout=corpus.software.holdout,
        identity=None,
        dry_run=dry_run,
        workdir=base / "generalist-software-exam",
        verbose=False,
    )

    plasma = run_specialist_loop(
        student_id=plasma_id.student_id,
        domain="plasma",
        train=list(corpus.plasma.train),
        holdout=list(corpus.plasma.holdout),
        identity=plasma_id,
        dry_run=dry_run,
        workdir=base / "plasma-specialist",
        verbose=True,
    )

    software = run_specialist_loop(
        student_id=software_id.student_id,
        domain="software",
        train=list(corpus.software.train),
        holdout=list(corpus.software.holdout),
        identity=software_id,
        dry_run=dry_run,
        workdir=base / "software-specialist",
        verbose=True,
    )

    comparison = {
        "variable": "dual_specialist_vs_generalist",
        "dry_run": dry_run,
        "plasma_holdout": {
            "generalist": _arm_dict(generalist),
            "specialist": _arm_dict(plasma),
            "delta_gain": plasma.last_score - generalist.last_score - (plasma.first_score - generalist.first_score),
        },
        "software_holdout": {
            "generalist": _arm_dict(generalist_sw),
            "specialist": _arm_dict(software),
            "delta_gain": software.last_score - generalist_sw.last_score - (software.first_score - generalist_sw.first_score),
        },
    }
    out = base / "dual_specialist_comparison.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n=== Dual-specialist comparison ===")
    print(f"  Plasma held-out:   generalist {generalist.first_score:.2f}->{generalist.last_score:.2f}  "
          f"specialist {plasma.first_score:.2f}->{plasma.last_score:.2f}")
    print(f"  Software held-out: generalist {generalist_sw.first_score:.2f}->{generalist_sw.last_score:.2f}  "
          f"specialist {software.first_score:.2f}->{software.last_score:.2f}")
    print(f"  Plasma contamination (specialist): {plasma.contamination:.0%}")
    print(f"  Software contamination (specialist): {software.contamination:.0%}")
    print(f"\nWrote {out}")
    print(f"Artifacts: {base}")


def _arm_dict(result: SpecialistRunResult) -> dict:
    return {
        "student_id": result.student_id,
        "first": result.first_score,
        "last": result.last_score,
        "peak": result.peak_score,
        "gain": result.last_score - result.first_score,
        "studied": result.samples_studied,
        "contamination": result.contamination,
    }


if __name__ == "__main__":
    main()
