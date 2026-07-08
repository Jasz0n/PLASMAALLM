"""Teacher visual brief review service (M31)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from allm.researcher.multimodal_types import DistilledVisualBrief, StudentVisualPackage, TeacherVisualApproval
from allm.researcher.student_visual_export import attach_student_visual_packages
from allm.researcher.types import KnowledgePackage
from allm.storage.base import RecordStore
from allm.teacher.visual_approval_store import VisualApprovalStore
from allm.teacher.visual_export import (
    VisualApprovalWorkflow,
    approve_visual_brief,
    export_student_visual,
)

ApprovalStatus = Literal["pending", "approved", "rejected"]


@dataclass(frozen=True)
class BriefRef:
    """One distilled brief with package context for Teacher review."""

    brief: DistilledVisualBrief
    package_id: str
    provider: str
    curriculum_topic: str | None


class TeacherVisualReviewService:
    """List, approve, and export distilled visual briefs for Teacher UI."""

    def __init__(self, store: RecordStore, packages: list[KnowledgePackage] | None = None) -> None:
        self._store = store
        self._packages = list(packages or [])
        self._workflow = VisualApprovalWorkflow(store)
        self._approval_store = VisualApprovalStore(store)

    def set_packages(self, packages: list[KnowledgePackage]) -> None:
        """Replace the in-memory package snapshot used for brief lookup."""
        self._packages = list(packages)

    def brief_refs(self) -> tuple[BriefRef, ...]:
        """All distilled briefs across stored packages."""
        refs: list[BriefRef] = []
        for package in self._packages:
            for brief in package.distilled_visual_briefs:
                refs.append(
                    BriefRef(
                        brief=brief,
                        package_id=package.id,
                        provider=package.provider,
                        curriculum_topic=package.curriculum_topic,
                    )
                )
        return tuple(refs)

    def approval_record(self, brief_id: str) -> TeacherVisualApproval | None:
        """Latest stored approval for one brief."""
        return self._approval_store.get(brief_id)

    def approval_status(self, brief_id: str) -> ApprovalStatus:
        """Resolve pending / approved / rejected for one brief."""
        stored = self._approval_store.get(brief_id)
        if stored is None:
            return "pending"
        return "approved" if stored.approved else "rejected"

    def list_briefs(
        self,
        *,
        source_kind: str | None = None,
        status: ApprovalStatus | Literal["all"] = "all",
    ) -> tuple[BriefRef, ...]:
        """Filter briefs by source kind and approval status."""
        rows = self.brief_refs()
        if source_kind:
            kind = source_kind.strip().lower()
            rows = [row for row in rows if row.brief.source_kind == kind]
        if status != "all":
            rows = [row for row in rows if self.approval_status(row.brief.brief_id) == status]
        return tuple(rows)

    def get_brief(self, brief_id: str) -> BriefRef | None:
        """Lookup one brief by id."""
        for row in self.brief_refs():
            if row.brief.brief_id == brief_id:
                return row
        return None

    def record_approval(
        self,
        brief_id: str,
        *,
        approved: bool,
        max_images: int = 2,
        max_questions: int = 3,
        include_diagram: bool = True,
        include_experiment: bool = True,
        approved_by: str = "teacher-ui",
        review_note: str = "",
    ) -> None:
        """Persist one Teacher decision."""
        ref = self.get_brief(brief_id)
        if ref is None:
            raise KeyError(f"unknown visual brief {brief_id!r}")
        approval = approve_visual_brief(
            ref.brief,
            approved=approved,
            max_images=max_images,
            max_questions=max_questions,
            include_diagram=include_diagram,
            include_experiment=include_experiment,
            approved_by=approved_by,
            review_note=review_note,
        )
        self._workflow.record(approval)

    def export_approved(self) -> tuple[StudentVisualPackage, ...]:
        """Export approved briefs and attach student packages to source packages."""
        refs = self.brief_refs()
        briefs = tuple(row.brief for row in refs)
        if not briefs:
            return ()

        approvals = self._workflow.resolve(briefs, auto_approve=False)
        approval_by_id = {approval.brief_id: approval for approval in approvals}

        exports_by_brief: dict[str, StudentVisualPackage] = {}
        for row in refs:
            approval = approval_by_id.get(row.brief.brief_id)
            if approval is None or not approval.approved:
                continue
            exported = export_student_visual(
                row.brief,
                approval,
                curriculum_topic=row.curriculum_topic or "",
            )
            if exported is not None:
                exports_by_brief[row.brief.brief_id] = exported

        updated_packages: list[KnowledgePackage] = []
        for package in self._packages:
            package_exports = [
                exports_by_brief[row.brief.brief_id]
                for row in refs
                if row.package_id == package.id and row.brief.brief_id in exports_by_brief
            ]
            if not package_exports:
                updated_packages.append(package)
                continue
            updated = package
            for export in package_exports:
                updated = attach_student_visual_packages(updated, (export,))
            updated_packages.append(updated)

        self._packages = updated_packages
        return tuple(exports_by_brief.values())

    def packages(self) -> tuple[KnowledgePackage, ...]:
        """Current package snapshot (including exported student visuals)."""
        return tuple(self._packages)

    def summary(self) -> dict[str, int]:
        """Counts for dashboard display."""
        refs = self.brief_refs()
        pending = sum(1 for row in refs if self.approval_status(row.brief.brief_id) == "pending")
        approved = sum(1 for row in refs if self.approval_status(row.brief.brief_id) == "approved")
        rejected = sum(1 for row in refs if self.approval_status(row.brief.brief_id) == "rejected")
        exports = sum(len(package.student_visual_packages) for package in self._packages)
        return {
            "total_briefs": len(refs),
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "student_exports": exports,
            "workshop_briefs": sum(1 for row in refs if row.brief.source_kind == "workshop"),
            "book_briefs": sum(1 for row in refs if row.brief.source_kind == "book"),
        }
