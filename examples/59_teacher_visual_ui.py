"""Teacher visual review UI demo — workshop + book selective approval (M31).

Runs a Researcher cycle (workshop + book figures, export deferred), then
exercises the Teacher HTTP API for selective approve/reject and export.

    PYTHONPATH=src python3 examples/59_teacher_visual_ui.py

Optional: start the API server after seeding:

    ALLM_STORAGE__PATH=/tmp/allm-teacher-ui.sqlite3 \\
    uvicorn --factory allm.api.app:create_default_app
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

try:
    from fastapi.testclient import TestClient
except ImportError as exc:
    raise SystemExit("Install api extras: pip install -e '.[api]'") from exc

from allm.api.app import create_app
from allm.core.logging import setup_logging  # noqa: E402
from allm.kdp.corpus import DEFAULT_TOPIC  # noqa: E402
from allm.researcher import ResearcherLayer  # noqa: E402
from allm.storage import SQLiteRecordStore  # noqa: E402


def _apply_defaults() -> None:
    defaults = {
        "ALLM_RESEARCHER": "1",
        "ALLM_MULTIMODAL": "1",
        "ALLM_BOOK_DISCOVERY": "1",
        "ALLM_BOOK_IMAGES": "1",
        "ALLM_VISION_CAPTIONS": "1",
        "ALLM_VISUAL_DISTILL": "1",
        "ALLM_VISUAL_EXPORT": "0",
        "ALLM_CROSS_SOURCE_VERIFY": "1",
        "ALLM_RESEARCHER_WORKSHOP_FILES": "2",
        "ALLM_BOOK_MAX_FILES": "3",
        "ALLM_BOOK_MAX_PAGES": "16",
        "ALLM_BOOK_MAX_IMAGES": "4",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    workdir = Path(tempfile.mkdtemp(prefix="allm-teacher-ui-"))
    db_path = workdir / "teacher_ui.sqlite3"
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(workdir / "book_images")

    print("\n=== M31 Phase 1: Researcher cycle (export deferred) ===")
    store = SQLiteRecordStore(db_path)
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        book_dir=Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books"))),
        workshop_max_files=int(os.environ.get("ALLM_RESEARCHER_WORKSHOP_FILES", "2")),
        book_max_files=int(os.environ.get("ALLM_BOOK_MAX_FILES", "3")),
        book_max_pages=int(os.environ.get("ALLM_BOOK_MAX_PAGES", "16")),
        book_max_images=int(os.environ.get("ALLM_BOOK_MAX_IMAGES", "4")),
        enable_book_images=True,
        book_images_cache_dir=workdir / "book_images",
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_captions=True,
        enable_visual_distillation=True,
        enable_visual_export=False,
        frames_cache_dir=workdir / "frames",
    )
    report = researcher.run_cycle()
    brief_count = sum(len(pkg.distilled_visual_briefs) for pkg in report.packages)
    print(f"  packages: {len(report.packages)}")
    print(f"  distilled briefs awaiting Teacher: {brief_count}")
    for package in report.packages:
        for brief in package.distilled_visual_briefs[:3]:
            print(
                f"    - [{brief.source_kind or 'unknown'}] {brief.brief_id}: "
                f"{brief.concept_name} conf={brief.evidence_confidence:.2f}"
            )
    store.close()

    print("\n=== M31 Phase 2: Teacher API selective review ===")
    app = create_app(db_path)
    with TestClient(app) as client:
        ui = client.get("/teacher/")
        assert ui.status_code == 200
        print("  UI route: GET /teacher/ OK")

        summary = client.get("/teacher/visual-review/summary").json()
        print(
            f"  pending={summary['pending']} workshop={summary['workshop_briefs']} "
            f"book={summary['book_briefs']}"
        )

        pending = client.get("/teacher/visual-briefs", params={"status": "pending"}).json()
        approved_count = 0
        for row in pending:
            approve = row["source_kind"] == "workshop" or row["evidence_confidence"] >= 0.75
            client.post(
                f"/teacher/visual-briefs/{row['brief_id']}/approve",
                json={
                    "approved": approve,
                    "approved_by": "teacher-ui-demo",
                    "review_note": "Approved for students" if approve else "Deferred",
                },
            )
            if approve:
                approved_count += 1
        print(f"  Teacher approved: {approved_count}/{len(pending)}")

        export = client.post("/teacher/visual-exports").json()
        print(f"  student visual exports: {export['export_count']}")
        for item in export["exports"][:5]:
            print(f"    export {item['export_id']}: {item['concept_name']}")

        final = client.get("/teacher/visual-review/summary").json()
        print(
            f"  final: approved={final['approved']} rejected={final['rejected']} "
            f"student_exports={final['student_exports']}"
        )

    print(f"\nDone. DB + artifacts under {workdir}")
    print(f"  Start UI: ALLM_STORAGE__PATH={db_path} uvicorn --factory allm.api.app:create_default_app")


if __name__ == "__main__":
    main()
