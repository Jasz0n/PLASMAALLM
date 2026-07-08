"""Bridge Teacher UI approvals into Researcher packages and KEL delivery (M32)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from allm.researcher.multimodal_types import DistilledVisualBrief, StudentVisualPackage
from allm.researcher.queue import RecommendationQueue
from allm.researcher.types import KnowledgePackage
from allm.storage.base import RecordStore
from allm.teacher.visual_export import approve_visual_brief
from allm.teacher.visual_review_service import TeacherVisualReviewService


@dataclass(frozen=True)
class TeacherApprovalPolicy:
    """Default selective approval thresholds for workshop and book briefs."""

    min_workshop_confidence: float = 0.75
    min_book_confidence: float = 0.75
    max_images: int = 2
    max_questions: int = 3
    approved_by: str = "teacher-selective"


@dataclass(frozen=True)
class TeacherVisualExportResult:
    """Outcome of exporting Teacher-approved briefs into student packages."""

    packages: tuple[KnowledgePackage, ...]
    exports: tuple[StudentVisualPackage, ...]
    approved_count: int
    rejected_count: int
    pending_count: int


def policy_from_env() -> TeacherApprovalPolicy:
    """Build approval policy from ALLM_TEACHER_* environment variables."""
    return TeacherApprovalPolicy(
        min_workshop_confidence=float(os.environ.get("ALLM_TEACHER_MIN_WORKSHOP_CONF", "0.75")),
        min_book_confidence=float(os.environ.get("ALLM_TEACHER_MIN_BOOK_CONF", "0.75")),
        max_images=int(os.environ.get("ALLM_VISUAL_EXPORT_IMAGES", "2")),
        max_questions=int(os.environ.get("ALLM_VISUAL_EXPORT_QUESTIONS", "3")),
        approved_by=os.environ.get("ALLM_TEACHER_APPROVED_BY", "teacher-selective"),
    )


def should_approve_brief(brief: DistilledVisualBrief, policy: TeacherApprovalPolicy) -> bool:
    """Return True when a brief meets the selective approval threshold."""
    if brief.source_kind == "book":
        return brief.evidence_confidence >= policy.min_book_confidence
    return brief.evidence_confidence >= policy.min_workshop_confidence


def apply_selective_approvals(
    service: TeacherVisualReviewService,
    policy: TeacherApprovalPolicy,
) -> tuple[int, int]:
    """Record approve/reject decisions for all pending briefs."""
    approved = rejected = 0
    for ref in service.list_briefs(status="pending"):
        approve = should_approve_brief(ref.brief, policy)
        service.record_approval(
            ref.brief.brief_id,
            approved=approve,
            max_images=min(policy.max_images, len(ref.brief.images)),
            max_questions=min(policy.max_questions, len(ref.brief.questions)),
            approved_by=policy.approved_by,
            review_note="Approved for student delivery" if approve else "Below Teacher threshold",
        )
        if approve:
            approved += 1
        else:
            rejected += 1
    return approved, rejected


def persist_teacher_packages(store: RecordStore, packages: tuple[KnowledgePackage, ...]) -> None:
    """Persist updated Knowledge Packages after Teacher export."""
    queue = RecommendationQueue(store)
    for package in packages:
        queue.store_package(package, reason="teacher visual export")


def export_teacher_approved(
    store: RecordStore,
    packages: list[KnowledgePackage] | tuple[KnowledgePackage, ...],
    *,
    policy: TeacherApprovalPolicy | None = None,
    apply_policy: bool = True,
) -> TeacherVisualExportResult:
    """Apply selective approvals (optional) and export student visual packages."""
    service = TeacherVisualReviewService(store, packages=list(packages))
    approved = rejected = 0
    if apply_policy and policy is not None:
        approved, rejected = apply_selective_approvals(service, policy)

    exports = service.export_approved()
    updated = service.packages()
    persist_teacher_packages(store, updated)
    summary = service.summary()
    return TeacherVisualExportResult(
        packages=updated,
        exports=exports,
        approved_count=summary["approved"],
        rejected_count=summary["rejected"],
        pending_count=summary["pending"],
    )


def sync_researcher_packages(researcher: object, packages: tuple[KnowledgePackage, ...]) -> None:
    """Push exported packages back into a ResearcherLayer snapshot."""
    if not hasattr(researcher, "persist_package"):
        return
    for package in packages:
        researcher.persist_package(package, reason="teacher UI export")
