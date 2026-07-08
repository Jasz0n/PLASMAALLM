"""Full-stack multimodal integration demo (M18).

Offline-friendly: fixtures + stub backends for vision/audio/OCR/LiveKit.
Wires Researcher (multimodal + livekit + archive) → dual KEL loop → debate
evidence → mediated consultation show-me.

    PYTHONPATH=src python3 examples/46_full_multimodal_stack.py
"""

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
    os.environ.setdefault("ALLM_DEBATE_EVIDENCE", "1")
    os.environ.setdefault("ALLM_CONSULT_SHOW_ME", "1")
    os.environ.setdefault("ALLM_MEDIATED_CONSULT", "1")
    os.environ.setdefault("ALLM_VISION_CAPTIONS", "1")
    os.environ.setdefault("ALLM_AUDIO_ANALYSIS", "1")
    os.environ.setdefault("ALLM_FRAME_OCR", "1")
    os.environ.setdefault("ALLM_VISION_ANALYTICS", "1")
    os.environ.setdefault("ALLM_MOTION_TRACKING", "1")
    os.environ.setdefault("ALLM_MOTION_CONTINUITY", "1")
    os.environ.setdefault("ALLM_OBJECT_IDENTITY", "1")
    os.environ.setdefault("ALLM_VISUAL_DISTILL", "1")
    os.environ.setdefault("ALLM_VISUAL_EXPORT", "1")
    os.environ.setdefault("ALLM_LIVEKIT", "1")
    os.environ.setdefault("ALLM_LIVEKIT_ARCHIVE", "1")
    os.environ.setdefault("ALLM_LIVEKIT_WORKER", "1")
    os.environ.setdefault("ALLM_ITERATIONS", "2")

    plasma = load_identity(STUDENTS / "plasma_student.yaml")
    software = load_identity(STUDENTS / "software_student.yaml")
    result = run_dual_mediated_loop(
        plasma_identity=plasma,
        software_identity=software,
        dry_run=True,
        workdir=Path(tempfile.mkdtemp(prefix="allm-full-stack-")),
    )

    print("\n=== M18: Full multimodal stack ===")
    print(f"  researcher packages: {result.researcher_packages}")
    print(f"  multimodal synced:   {result.multimodal_synced}")
    print(f"  live evidence:       {result.live_evidence_count}")
    print(f"  worker streams:      {result.worker_streams}")
    print(f"  archived fixtures:   {result.archived_fixtures}")
    print(f"  software last score: {result.software_score_last:.2f}")
    print(f"  plasma last score:   {result.plasma_score_last:.2f}")
    print(f"  mediated approvals:  {result.mediated_approvals}")
    print(f"  debate evidence hits:{result.debate_evidence_hits}")
    print(f"\nDone. Artifacts under {result.workdir}")


if __name__ == "__main__":
    main()
