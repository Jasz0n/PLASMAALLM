"""Tests for Teacher visual → KEL bridge (M32)."""

from allm.researcher.multimodal_types import DistilledVisualBrief
from allm.researcher.types import KnowledgePackage
from allm.storage import SQLiteRecordStore
from allm.teacher.visual_kel_bridge import (
    TeacherApprovalPolicy,
    export_teacher_approved,
    should_approve_brief,
)
from allm.teacher.visual_review_service import TeacherVisualReviewService


def _brief(brief_id: str, *, source_kind: str, confidence: float) -> DistilledVisualBrief:
    return DistilledVisualBrief(
        brief_id=brief_id,
        concept_name=f"concept-{brief_id}",
        images=("visual",),
        evidence_confidence=confidence,
        source_kind=source_kind,
    )


def test_should_approve_brief_by_source_threshold() -> None:
    policy = TeacherApprovalPolicy(min_workshop_confidence=0.8, min_book_confidence=0.7)
    assert should_approve_brief(_brief("a", source_kind="workshop", confidence=0.85), policy)
    assert not should_approve_brief(_brief("b", source_kind="workshop", confidence=0.7), policy)
    assert should_approve_brief(_brief("c", source_kind="book", confidence=0.72), policy)


def test_export_teacher_approved_persists_student_packages() -> None:
    store = SQLiteRecordStore(":memory:")
    packages = [
        KnowledgePackage.build(
            provider="kids-workshops",
            title="Workshop",
            curriculum_topic="kids-plasma",
            distilled_visual_briefs=(
                _brief("dvis_ok", source_kind="workshop", confidence=0.9),
                _brief("dvis_no", source_kind="workshop", confidence=0.5),
            ),
        )
    ]
    result = export_teacher_approved(
        store,
        packages,
        policy=TeacherApprovalPolicy(min_workshop_confidence=0.75),
    )
    assert result.approved_count == 1
    assert result.rejected_count == 1
    assert len(result.exports) == 1
    assert len(result.packages[0].student_visual_packages) == 1

    service = TeacherVisualReviewService(store, packages=list(result.packages))
    assert service.summary()["student_exports"] == 1
