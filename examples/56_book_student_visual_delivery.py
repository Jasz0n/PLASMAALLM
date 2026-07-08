"""Book visual briefs → Teacher approval → student delivery demo (M28).

Full book path: PDF figures, vision enrichment, per-page briefs,
selective Teacher approval, and student-safe visual packages.

    PYTHONPATH=src python3 examples/56_book_student_visual_delivery.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.models.base import ModelSpec
from allm.models.echo import EchoModel
from allm.researcher import ResearcherLayer
from allm.storage import SQLiteRecordStore
from allm.students.model_student import ModelStudent, ModelStudentConfig
from allm.teacher.student_visual_delivery import deliver_visual_notes
from allm.teacher.visual_export import VisualApprovalWorkflow, approve_visual_brief

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    books_dir = ROOT / "books"
    if not books_dir.is_dir():
        raise SystemExit(f"Book directory not found: {books_dir}")

    workdir = Path(tempfile.mkdtemp(prefix="allm-book-student-visual-"))
    store = SQLiteRecordStore(workdir / "delivery.sqlite3")

    max_pages = int(os.environ.get("ALLM_BOOK_MAX_PAGES", "40"))
    max_images = int(os.environ.get("ALLM_BOOK_MAX_IMAGES", "8"))
    auto_approve = os.environ.get("ALLM_VISUAL_EXPORT_AUTO", "0") == "1"

    researcher = ResearcherLayer(
        store,
        workshop_max_files=0,
        book_dir=books_dir,
        book_max_files=int(os.environ.get("ALLM_BOOK_MAX_FILES", "1")),
        book_max_pages=max_pages,
        book_max_images=max_images,
        enable_book_images=True,
        book_images_cache_dir=workdir / "book_images",
        enable_vision_captions=os.environ.get("ALLM_VISION_CAPTIONS", "1") == "1",
        enable_frame_ocr=os.environ.get("ALLM_FRAME_OCR", "1") == "1",
        enable_visual_distillation=True,
        enable_visual_export=True,
        visual_export_auto_approve=auto_approve,
        visual_export_min_confidence=float(os.environ.get("ALLM_VISUAL_EXPORT_MIN_CONF", "0.75")),
        catalog_topics=(DEFAULT_TOPIC,),
    )

    if not auto_approve:
        report = researcher.run_cycle()
        book_packages = [pkg for pkg in report.packages if pkg.provider == "keshe-books"]
        all_briefs = tuple(
            brief for package in book_packages for brief in package.distilled_visual_briefs
        )
        print(f"\n=== M28 Phase 1: {len(all_briefs)} book brief(s) awaiting Teacher review ===")
        workflow = VisualApprovalWorkflow(store)
        for brief in all_briefs:
            approve = brief.source_kind == "book" and brief.evidence_confidence >= 0.75
            workflow.record(
                approve_visual_brief(
                    brief,
                    approved=approve,
                    max_images=min(2, len(brief.images)),
                    max_questions=min(2, len(brief.questions)),
                    approved_by="teacher-book-selective",
                    review_note="Book diagram approved for students" if approve else "Deferred",
                )
            )
            print(f"  {brief.brief_id}: {brief.concept_name} -> {'approved' if approve else 'rejected'}")

        from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
        from allm.researcher.capabilities.visual_export import VisualExportCapability
        from allm.researcher.capabilities.verification import GraphVerificationCapability

        ctx = CapabilityContext(
            store=store,
            config=ResearcherPipelineConfig(
                book_dir=books_dir,
                book_curriculum_topic=DEFAULT_TOPIC,
                enable_visual_export=True,
                visual_export_auto_approve=False,
                visual_export_persist_approvals=True,
            ),
        )
        pipeline = PipelineState()
        pipeline.packages = list(book_packages)
        GraphVerificationCapability().run(ctx, pipeline)
        export_result = VisualExportCapability().run(ctx, pipeline)
        book_packages = list(pipeline.verified_packages or pipeline.packages)
        for package in book_packages:
            researcher.persist_package(package, reason="book visual export")
        report = researcher.last_report()
    else:
        report = researcher.run_cycle()
        book_packages = [pkg for pkg in report.packages if pkg.provider == "keshe-books"]

    print("\n=== M28 Phase 2: student-safe book visual exports ===")
    export_count = 0
    for package in book_packages:
        for export in package.student_visual_packages:
            export_count += 1
            print(f"  export {export.export_id}: {export.concept_name}")
            print(f"    images: {len(export.images)} questions: {len(export.questions)}")
            print(f"    approved_by: {export.approved_by}")

    print("\n=== M28 Phase 3: deliver into student study memory ===")
    student = ModelStudent(
        "kids-book-demo",
        DEFAULT_TOPIC,
        EchoModel(ModelSpec(name="demo", provider="echo", model_id="none")),
        ModelStudentConfig(max_notes=128),
    )
    stored_exports = researcher.student_visual_packages(topic=DEFAULT_TOPIC)
    delivered = deliver_visual_notes(student, stored_exports)
    print(f"  visual study notes delivered: {delivered}")
    print(f"  student notes: {len(student.notes)}")

    for name, yield_count, notes in (report.capability_summary if report else ()):
        if name in {
            "understanding.book.images",
            "understanding.visual.distill",
            "understanding.visual.export",
        }:
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
