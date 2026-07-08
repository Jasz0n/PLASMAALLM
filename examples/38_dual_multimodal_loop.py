"""Example 29 wrapper — dual specialist with multimodal debate evidence."""

from __future__ import annotations

import os
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
    os.environ.setdefault("ALLM_RESEARCHER", "1")
    os.environ.setdefault("ALLM_MULTIMODAL", "1")
    os.environ.setdefault("ALLM_CONSULT_SHOW_ME", "1")
    os.environ.setdefault("ALLM_MEDIATED_CONSULT", "1")

    plasma = load_identity(STUDENTS / "plasma_student.yaml")
    software = load_identity(STUDENTS / "software_student.yaml")
    result = run_dual_mediated_loop(
        plasma_identity=plasma,
        software_identity=software,
        dry_run=True,
        workdir=Path(tempfile.mkdtemp(prefix="allm-dual-multimodal-")),
    )
    print(f"\n  software last score: {result.software_score_last:.2f}")
    print(f"  plasma last score:   {result.plasma_score_last:.2f}")
    print(f"  mediated approvals:  {result.mediated_approvals}")
    print(f"  researcher packages: {result.researcher_packages}")
    print(f"  debate evidence hits:{result.debate_evidence_hits}")
    print(f"\nDone. Artifacts under {result.workdir}")


if __name__ == "__main__":
    main()
