"""Dual-specialist loop with Teacher-mediated consultation (offline).

    PYTHONPATH=src python3 examples/29_mediated_dual_loop.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.core.logging import setup_logging
from allm.students import load_identity
from dual_consult_run import run_dual_mediated_loop

STUDENTS = ROOT / "configs/students"


def main() -> None:
    setup_logging("INFO")
    plasma = load_identity(STUDENTS / "plasma_student.yaml")
    software = load_identity(STUDENTS / "software_student.yaml")
    result = run_dual_mediated_loop(
        plasma_identity=plasma,
        software_identity=software,
        dry_run=True,
        workdir=Path(tempfile.mkdtemp(prefix="allm-mediated-dual-")),
    )
    print(f"\n  software last score: {result.software_score_last:.2f}")
    print(f"  plasma last score:   {result.plasma_score_last:.2f}")
    print(f"  mediated approvals logged: {result.mediated_approvals}")
    print(f"\nDone. Artifacts under {result.workdir}")


if __name__ == "__main__":
    main()
