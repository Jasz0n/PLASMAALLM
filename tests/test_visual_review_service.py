"""Tests for Teacher visual review service (M31)."""

from allm.researcher.multimodal_types import DistilledVisualBrief
from allm.researcher.types import KnowledgePackage
from allm.storage import SQLiteRecordStore
from allm.teacher.visual_review_service import TeacherVisualReviewService


def _brief(brief_id: str, *, source_kind: str = "workshop") -> DistilledVisualBrief:
    return DistilledVisualBrief(
        brief_id=brief_id,
        concept_name=f"concept-{brief_id}",
        concept_description="Teacher-facing description",
        images=("Magnet rotation demo",),
        diagram_summary="Field between poles",
        evidence_confidence=0.85,
        source_kind=source_kind,
        teacher_notes="Internal teacher note",
    )


def _package(briefs: tuple[DistilledVisualBrief, ...], *, provider: str) -> KnowledgePackage:
    return KnowledgePackage.build(
        provider=provider,
        title=f"{provider} package",
        curriculum_topic="kids-plasma",
        distilled_visual_briefs=briefs,
    )


def test_list_briefs_filters_by_source_and_status() -> None:
    store = SQLiteRecordStore(":memory:")
    workshop = _package((_brief("dvis_w", source_kind="workshop"),), provider="kids-workshops")
    book = _package((_brief("dvis_b", source_kind="book"),), provider="keshe-books")
    service = TeacherVisualReviewService(store, packages=[workshop, book])

    assert len(service.list_briefs(source_kind="book")) == 1
    assert len(service.list_briefs(status="pending")) == 2

    service.record_approval("dvis_w", approved=True)
    assert len(service.list_briefs(status="approved")) == 1
    assert len(service.list_briefs(status="pending")) == 1


def test_export_approved_attaches_student_packages() -> None:
    store = SQLiteRecordStore(":memory:")
    high = _brief("dvis_ok", source_kind="workshop")
    low = _brief("dvis_no", source_kind="book")
    package = _package((high, low), provider="mixed")
    service = TeacherVisualReviewService(store, packages=[package])

    service.record_approval("dvis_ok", approved=True, max_images=1)
    service.record_approval("dvis_no", approved=False, review_note="Low confidence")

    exports = service.export_approved()
    assert len(exports) == 1
    assert exports[0].concept_name == "concept-dvis_ok"

    updated = service.packages()[0]
    assert len(updated.student_visual_packages) == 1
    assert updated.student_visual_packages[0].concept_name == "concept-dvis_ok"
