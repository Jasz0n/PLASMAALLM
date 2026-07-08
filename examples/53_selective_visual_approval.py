"""Selective Teacher visual approval + KEL delivery demo (M25).

Teacher reviews distilled briefs individually, persists approvals,
exports only approved subsets, then delivers visual study notes into
the learning loop — students never receive raw video.

    PYTHONPATH=src python3 examples/53_selective_visual_approval.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.student_visual_export import attach_student_visual_packages
from allm.storage import SQLiteRecordStore
from allm.teacher.student_visual_delivery import deliver_visual_notes
from allm.teacher.visual_export import (
    VisualApprovalWorkflow,
    approve_visual_brief,
    export_approved_briefs,
)
from dual_consult_run import run_dual_mediated_loop
from allm.students import load_identity

STUDENTS = ROOT / "configs/students"


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-selective-visual-"))
    store = SQLiteRecordStore(workdir / "approval.sqlite3")

    print("\n=== M25 Phase 1: distill briefs (export deferred) ===")
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_analytics=True,
        enable_motion_tracking=True,
        enable_motion_continuity=True,
        enable_object_identity=True,
        enable_visual_distillation=True,
        enable_visual_export=False,
        frames_cache_dir=workdir / "frames",
    )
    report = researcher.run_cycle()
    all_briefs = tuple(
        brief
        for package in report.packages
        for brief in package.distilled_visual_briefs
    )
    print(f"  distilled briefs: {len(all_briefs)}")
    for brief in all_briefs:
        print(f"    - {brief.brief_id}: {brief.concept_name} conf={brief.evidence_confidence:.2f}")

    print("\n=== M25 Phase 2: Teacher selective approval ===")
    workflow = VisualApprovalWorkflow(store)
    pending = workflow.pending_briefs(all_briefs)
    print(f"  pending review: {len(pending)}")
    approved_count = 0
    for brief in all_briefs:
        approve = brief.evidence_confidence >= float(
            os.environ.get("ALLM_VISUAL_EXPORT_MIN_CONF", "0.7")
        )
        if approve:
            approved_count += 1
        workflow.record(
            approve_visual_brief(
                brief,
                approved=approve,
                max_images=int(os.environ.get("ALLM_VISUAL_EXPORT_IMAGES", "2")),
                max_questions=int(os.environ.get("ALLM_VISUAL_EXPORT_QUESTIONS", "2")),
                approved_by="teacher-selective" if approve else "teacher-rejected",
                review_note="Approved for student delivery" if approve else "Below confidence threshold",
            )
        )
    print(f"  Teacher approved: {approved_count}/{len(all_briefs)}")

    print("\n=== M25 Phase 3: export approved subsets ===")
    approvals = workflow.resolve(all_briefs, auto_approve=False)
    exports = export_approved_briefs(
        all_briefs,
        approvals,
        curriculum_topic=DEFAULT_TOPIC,
    )
    print(f"  student visual packages: {len(exports)}")
    for export in exports:
        print(f"    export {export.export_id}: {export.concept_name} ({len(export.images)} images)")

    if report.packages and exports:
        package = report.packages[0]
        updated = attach_student_visual_packages(package, exports)
        researcher.persist_package(updated, reason="selective export")

    print("\n=== M25 Phase 4: deliver into student study memory ===")
    from allm.students.model_student import ModelStudent, ModelStudentConfig
    from allm.models.echo import EchoModel
    from allm.models.base import ModelSpec

    student = ModelStudent(
        "kids-demo",
        DEFAULT_TOPIC,
        EchoModel(ModelSpec(name="demo", provider="echo", model_id="none")),
        ModelStudentConfig(max_notes=64),
    )
    stored_exports = researcher.student_visual_packages(topic=DEFAULT_TOPIC)
    delivered = deliver_visual_notes(student, stored_exports)
    print(f"  visual study notes delivered: {delivered}")
    print(f"  student notes after delivery: {len(student.notes)}")

    print("\n=== M25 Phase 5: capstone loop with visual delivery ===")
    os.environ.setdefault("ALLM_RESEARCHER", "1")
    os.environ.setdefault("ALLM_MULTIMODAL", "1")
    os.environ.setdefault("ALLM_VISUAL_DISTILL", "1")
    os.environ.setdefault("ALLM_VISUAL_EXPORT", "1")
    os.environ.setdefault("ALLM_VISUAL_EXPORT_AUTO", "0")
    os.environ.setdefault("ALLM_VISUAL_DELIVERY", "1")
    os.environ.setdefault("ALLM_ITERATIONS", "1")

    plasma = load_identity(STUDENTS / "plasma_student.yaml")
    software = load_identity(STUDENTS / "software_student.yaml")
    loop_result = run_dual_mediated_loop(
        plasma_identity=plasma,
        software_identity=software,
        dry_run=True,
        workdir=workdir / "loop",
        verbose=False,
    )
    print(f"  loop researcher packages: {loop_result.researcher_packages}")
    print(f"  plasma score: {loop_result.plasma_score_last:.2f}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
