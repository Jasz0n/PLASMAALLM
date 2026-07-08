"""Student-safe visual export demo (M24).

Teacher approves a subset of distilled visual briefs; students receive
text descriptions, diagrams, explanations, and questions — never raw video.

    PYTHONPATH=src python3 examples/52_student_visual_export.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.storage import SQLiteRecordStore
from allm.teacher.visual_export import approve_visual_brief, export_student_visual

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-student-visual-export-"))
    store = SQLiteRecordStore(workdir / "export.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_analytics=os.environ.get("ALLM_VISION_ANALYTICS", "1") == "1",
        enable_motion_tracking=os.environ.get("ALLM_MOTION_TRACKING", "1") == "1",
        enable_motion_continuity=os.environ.get("ALLM_MOTION_CONTINUITY", "1") == "1",
        enable_object_identity=os.environ.get("ALLM_OBJECT_IDENTITY", "1") == "1",
        enable_visual_distillation=os.environ.get("ALLM_VISUAL_DISTILL", "1") == "1",
        enable_visual_export=True,
        visual_export_auto_approve=os.environ.get("ALLM_VISUAL_EXPORT_AUTO", "1") == "1",
        visual_export_min_confidence=float(os.environ.get("ALLM_VISUAL_EXPORT_MIN_CONF", "0.7")),
        visual_export_max_images=int(os.environ.get("ALLM_VISUAL_EXPORT_IMAGES", "2")),
        visual_export_max_questions=int(os.environ.get("ALLM_VISUAL_EXPORT_QUESTIONS", "3")),
        frames_cache_dir=workdir / "frames",
    )
    report = researcher.run_cycle()

    print("\n=== M24: Student-safe visual export ===")
    export_count = 0
    for package in report.packages:
        if not package.student_visual_packages:
            continue
        print(f"\n  package: {package.id}")
        for export in package.student_visual_packages:
            export_count += 1
            print(f"\n    export: {export.export_id}")
            print(f"      concept: {export.concept_name}")
            print(f"      approved_by: {export.approved_by}")
            print(f"      images: {len(export.images)}")
            for image in export.images:
                print(f"        - {image[:90]}...")
            if export.diagram:
                print(f"      diagram: {export.diagram[:90]}...")
            print(f"      explanations: {len(export.explanations)}")
            if export.experiment:
                print(f"      experiment: {export.experiment[:90]}...")
            print(f"      questions: {len(export.questions)}")
            for question in export.questions:
                print(f"        ? {question}")

    print(f"\n  student visual packages: {export_count}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.visual.export":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    if report.packages and report.packages[0].distilled_visual_briefs:
        brief = report.packages[0].distilled_visual_briefs[0]
        manual = export_student_visual(
            brief,
            approve_visual_brief(brief, max_images=1, max_questions=2, approved_by="teacher-manual"),
            curriculum_topic=DEFAULT_TOPIC,
        )
        if manual is not None:
            print(f"\n  manual Teacher approval demo: {manual.export_id} ({len(manual.images)} images)")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
