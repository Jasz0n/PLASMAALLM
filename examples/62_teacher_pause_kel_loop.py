"""Teacher UI pause → review → resume KEL loop demo (M34).

Runs Researcher distill, pauses for Teacher review, then continues the
held-out KEL loop with visual delivery.

    # CI-friendly simulate (background Teacher approval)
    PYTHONPATH=src python3 examples/62_teacher_pause_kel_loop.py --simulate-teacher

    # Manual: pause until resume flag is touched
    PYTHONPATH=src python3 examples/62_teacher_pause_kel_loop.py
    # In another terminal: touch <workdir>/teacher_resume.flag
    # Or start API and use GET /teacher/ then Export + Resume
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.core.logging import setup_logging
from allm.researcher.queue import RecommendationQueue
from allm.teacher.teacher_kel_session import TeacherKelSessionStore
from allm.teacher.visual_kel_bridge import export_teacher_approved, policy_from_env
from kids_kel_run import run_kids_kel_steered


def _apply_defaults() -> None:
    defaults = {
        "ALLM_RESEARCHER": "1",
        "ALLM_MULTIMODAL": "1",
        "ALLM_BOOK_DISCOVERY": "1",
        "ALLM_BOOK_IMAGES": "1",
        "ALLM_VISION_CAPTIONS": "1",
        "ALLM_VISUAL_DISTILL": "1",
        "ALLM_CROSS_SOURCE_VERIFY": "1",
        "ALLM_TEACHER_UI_PAUSE": "1",
        "ALLM_VISUAL_DELIVERY": "1",
        "ALLM_ITERATIONS": "2",
        "ALLM_RESEARCHER_WORKSHOP_FILES": "2",
        "ALLM_BOOK_MAX_FILES": "1",
        "ALLM_BOOK_MAX_PAGES": "16",
        "ALLM_BOOK_MAX_IMAGES": "4",
        "ALLM_BOOTSTRAP": "1",
        "ALLM_TEACHER_PAUSE_POLL": "0.5",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _simulate_teacher_review(workdir: Path, delay: float = 2.0) -> None:
    """Background thread: wait for pause session, approve, export, resume."""
    from allm.storage import SQLiteRecordStore

    time.sleep(delay)
    db_path = workdir / "kids.sqlite3"
    for _ in range(120):
        if db_path.is_file():
            break
        time.sleep(0.5)
    if not db_path.is_file():
        return

    store = SQLiteRecordStore(db_path)
    session_store = TeacherKelSessionStore(store)
    for _ in range(60):
        if session_store.get() is not None:
            break
        time.sleep(0.5)

    packages = RecommendationQueue(store).packages()
    if not packages:
        store.close()
        return

    result = export_teacher_approved(store, packages, policy=policy_from_env())
    session_store.mark_exported(student_exports=len(result.exports))
    run_dir = session_store.get().run_dir if session_store.get() else None
    if run_dir:
        Path(run_dir).joinpath("teacher_resume.flag").touch()
    store.close()


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="Teacher pause/resume KEL capstone (M34)")
    parser.add_argument(
        "--simulate-teacher",
        action="store_true",
        help="auto-approve and resume in background (CI-friendly)",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="allm-teacher-pause-"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")

    if args.simulate_teacher:
        threading.Thread(
            target=_simulate_teacher_review,
            args=(workdir,),
            daemon=True,
        ).start()

    print("\n=== M34: Teacher pause → resume KEL capstone ===")
    result = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=workdir,
        verbose=True,
    )

    print("\n=== Capstone summary ===")
    print(f"  teacher paused:        {result.teacher_paused}")
    print(f"  Teacher approved:      {result.teacher_approved_briefs}")
    print(f"  student visual exports:{result.student_visual_exports}")
    print(f"  visual notes delivered:  {result.visual_notes_delivered}")
    print(f"  exam score:            {result.first_score:.2f} -> {result.last_score:.2f}")
    print(f"\nDone. Artifacts under {result.workdir}")
    print(f"  DB for API: ALLM_STORAGE__PATH={result.workdir / 'kids.sqlite3'}")


if __name__ == "__main__":
    main()
