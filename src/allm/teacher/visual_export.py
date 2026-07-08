"""Teacher-approved export of distilled visuals for student delivery."""

from __future__ import annotations

from allm.kdp.types import content_hash
from allm.researcher.multimodal_types import (
    DistilledVisualBrief,
    StudentVisualPackage,
    TeacherVisualApproval,
)
from allm.storage.base import RecordStore
from allm.teacher.visual_approval_store import VisualApprovalStore


def approve_visual_brief(
    brief: DistilledVisualBrief,
    *,
    approved: bool = True,
    max_images: int = 2,
    max_questions: int = 3,
    include_diagram: bool = True,
    include_experiment: bool = True,
    approved_by: str = "teacher",
    review_note: str = "",
) -> TeacherVisualApproval:
    """Record Teacher approval settings for one distilled visual brief."""
    return TeacherVisualApproval(
        brief_id=brief.brief_id,
        approved=approved,
        max_images=max(0, min(max_images, len(brief.images))),
        max_questions=max(0, min(max_questions, len(brief.questions))),
        include_diagram=include_diagram and brief.diagram_summary is not None,
        include_experiment=include_experiment and brief.experiment_prompt is not None,
        approved_by=approved_by,
        review_note=review_note,
    )


def export_student_visual(
    brief: DistilledVisualBrief,
    approval: TeacherVisualApproval,
    *,
    curriculum_topic: str = "",
) -> StudentVisualPackage | None:
    """Export a Teacher-approved subset of one brief for student delivery."""
    if not approval.approved or approval.brief_id != brief.brief_id:
        return None

    images = brief.images[: approval.max_images]
    questions = brief.questions[: approval.max_questions]
    explanations = brief.explanations[:2]
    diagram = brief.diagram_summary if approval.include_diagram else None
    experiment = brief.experiment_prompt if approval.include_experiment else None

    export_id = "svis_" + content_hash(brief.brief_id, approval.approved_by, curriculum_topic)
    return StudentVisualPackage(
        export_id=export_id,
        concept_name=brief.concept_name,
        concept_description=brief.concept_description,
        images=images,
        diagram=diagram,
        explanations=explanations,
        experiment=experiment,
        questions=questions,
        curriculum_topic=curriculum_topic,
        approved_by=approval.approved_by,
    )


def auto_approve_briefs(
    briefs: tuple[DistilledVisualBrief, ...],
    *,
    min_confidence: float = 0.7,
    max_images: int = 2,
    max_questions: int = 3,
    approved_by: str = "teacher-auto",
) -> tuple[TeacherVisualApproval, ...]:
    """Stub auto-approval for dev/CI when brief confidence meets threshold."""
    approvals: list[TeacherVisualApproval] = []
    for brief in briefs:
        approved = brief.evidence_confidence >= min_confidence
        approvals.append(
            approve_visual_brief(
                brief,
                approved=approved,
                max_images=max_images,
                max_questions=max_questions,
                approved_by=approved_by if approved else "teacher-rejected",
            )
        )
    return tuple(approvals)


def export_approved_briefs(
    briefs: tuple[DistilledVisualBrief, ...],
    approvals: tuple[TeacherVisualApproval, ...],
    *,
    curriculum_topic: str = "",
) -> tuple[StudentVisualPackage, ...]:
    """Export all approved briefs into student-safe visual packages."""
    approval_by_id = {approval.brief_id: approval for approval in approvals}
    exports: list[StudentVisualPackage] = []
    for brief in briefs:
        approval = approval_by_id.get(brief.brief_id)
        if approval is None:
            continue
        exported = export_student_visual(
            brief,
            approval,
            curriculum_topic=curriculum_topic,
        )
        if exported is not None:
            exports.append(exported)
    return tuple(exports)


def resolve_visual_approvals(
    briefs: tuple[DistilledVisualBrief, ...],
    *,
    store: RecordStore | None = None,
    auto_approve: bool = True,
    min_confidence: float = 0.7,
    max_images: int = 2,
    max_questions: int = 3,
    persist: bool = True,
) -> tuple[TeacherVisualApproval, ...]:
    """Resolve approvals from store, auto-approval, or pending rejection."""
    approval_store = VisualApprovalStore(store) if store is not None else None
    stored = {row.brief_id: row for row in approval_store.all_approvals()} if approval_store else {}
    resolved: list[TeacherVisualApproval] = []

    for brief in briefs:
        existing = stored.get(brief.brief_id)
        if existing is not None:
            resolved.append(existing)
            continue

        if auto_approve:
            approval = approve_visual_brief(
                brief,
                approved=brief.evidence_confidence >= min_confidence,
                max_images=max_images,
                max_questions=max_questions,
                approved_by="teacher-auto" if brief.evidence_confidence >= min_confidence else "teacher-rejected",
            )
            if approval_store is not None and persist:
                approval_store.save(approval, reason="auto")
            resolved.append(approval)

    return tuple(resolved)


class VisualApprovalWorkflow:
    """Teacher selective review workflow for distilled visual briefs."""

    def __init__(self, store: RecordStore) -> None:
        self._record_store = store
        self._store = VisualApprovalStore(store)

    def pending_briefs(
        self,
        briefs: tuple[DistilledVisualBrief, ...],
    ) -> tuple[DistilledVisualBrief, ...]:
        """Briefs awaiting a Teacher decision."""
        return self._store.pending_briefs(briefs)

    def record(self, approval: TeacherVisualApproval) -> None:
        """Persist one Teacher approval or rejection."""
        self._store.save(approval, reason="teacher selective review")

    def resolve(
        self,
        briefs: tuple[DistilledVisualBrief, ...],
        *,
        auto_approve: bool = False,
        min_confidence: float = 0.7,
        max_images: int = 2,
        max_questions: int = 3,
    ) -> tuple[TeacherVisualApproval, ...]:
        """Resolve approvals using stored decisions and optional auto-approval."""
        return resolve_visual_approvals(
            briefs,
            store=self._record_store,
            auto_approve=auto_approve,
            min_confidence=min_confidence,
            max_images=max_images,
            max_questions=max_questions,
            persist=True,
        )
