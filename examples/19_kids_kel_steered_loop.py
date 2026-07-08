"""Kids plasma: KEL-steered held-out learning loop (M4).

    PYTHONPATH=src python3 examples/19_kids_kel_steered_loop.py

Disable mission: ``ALLM_STUDENT_IDENTITY=0``
Fixed exams across runs: ``ALLM_LOOP_SEED=42``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.core.logging import setup_logging
from kids_kel_run import run_kids_kel_steered


def main() -> None:
    setup_logging("INFO")
    identity_path = os.environ.get("ALLM_STUDENT_IDENTITY", "configs/students/kids_kel_plasma.yaml")
    result = run_kids_kel_steered(identity_path=identity_path, verbose=True)
    print(
        f"\nDone. holdout {result.first_score:.2f} -> {result.last_score:.2f} "
        f"(peak {result.peak_score:.2f}) mission={'on' if result.mission_enabled else 'off'}"
    )
    print(f"Artifacts under {result.workdir}")


if __name__ == "__main__":
    main()
