"""Teacher-approved visuals → KEL capstone loop (M32).

Researcher distills workshop + book briefs with export deferred, Teacher
selective approval runs via the KEL bridge, then the held-out loop delivers
approved visual study notes into student memory.

    PYTHONPATH=src python3 examples/60_teacher_approved_kel_loop.py

Environment (defaults enable Teacher UI approval path):
    ALLM_RESEARCHER=1
    ALLM_TEACHER_UI_APPROVAL=1
    ALLM_VISUAL_DELIVERY=1
    ALLM_ITERATIONS=2
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.core.logging import setup_logging
from kids_kel_run import run_kids_kel_steered


def _apply_defaults() -> None:
    defaults = {
        "ALLM_RESEARCHER": "1",
        "ALLM_MULTIMODAL": "1",
        "ALLM_BOOK_DISCOVERY": "1",
        "ALLM_BOOK_IMAGES": "1",
        "ALLM_VISION_CAPTIONS": "1",
        "ALLM_FRAME_OCR": "1",
        "ALLM_VISION_ANALYTICS": "1",
        "ALLM_MOTION_TRACKING": "1",
        "ALLM_MOTION_CONTINUITY": "1",
        "ALLM_OBJECT_IDENTITY": "1",
        "ALLM_VISUAL_DISTILL": "1",
        "ALLM_CROSS_SOURCE_VERIFY": "1",
        "ALLM_TEACHER_UI_APPROVAL": "1",
        "ALLM_TEACHER_MIN_WORKSHOP_CONF": "0.75",
        "ALLM_TEACHER_MIN_BOOK_CONF": "0.75",
        "ALLM_VISUAL_DELIVERY": "1",
        "ALLM_ITERATIONS": "2",
        "ALLM_RESEARCHER_WORKSHOP_FILES": "2",
        "ALLM_BOOK_MAX_FILES": "3",
        "ALLM_BOOK_MAX_PAGES": "24",
        "ALLM_BOOK_MAX_IMAGES": "6",
        "ALLM_BOOTSTRAP": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    workdir = Path(tempfile.mkdtemp(prefix="allm-teacher-kel-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")

    print("\n=== M32: Teacher-approved workshop + book KEL capstone ===")
    result = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=workdir,
        verbose=True,
    )

    print("\n=== Capstone summary ===")
    print(f"  Teacher approved briefs: {result.teacher_approved_briefs}")
    print(f"  Teacher rejected briefs: {result.teacher_rejected_briefs}")
    print(f"  student visual exports:  {result.student_visual_exports}")
    print(f"  visual notes delivered:  {result.visual_notes_delivered}")
    print(f"  workshop packages:       {result.workshop_packages}")
    print(f"  book packages:           {result.book_packages}")
    print(f"  cross-source aligned:    {result.aligned_concepts}")
    print(f"  KEL learning gain:       {result.kel_lg:.3f}" if result.kel_lg is not None else "  KEL learning gain:       n/a")
    print(f"  exam score:              {result.first_score:.2f} -> {result.last_score:.2f} (peak {result.peak_score:.2f})")
    print(f"  iterations:              {result.iterations_completed}")
    print(f"\nDone. Artifacts under {result.workdir}")


if __name__ == "__main__":
    main()
