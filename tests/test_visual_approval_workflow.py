"""Tests for Teacher selective visual approval workflow (M25)."""

from pathlib import Path

from allm.researcher.multimodal_types import DistilledVisualBrief
from allm.storage import SQLiteRecordStore
from allm.teacher.visual_approval_store import VisualApprovalStore
from allm.teacher.visual_export import (
    VisualApprovalWorkflow,
    approve_visual_brief,
    export_approved_briefs,
    resolve_visual_approvals,
)

ROOT = Path(__file__).resolve().parents[1]


def _brief(brief_id: str, *, confidence: float = 0.9) -> DistilledVisualBrief:
    return DistilledVisualBrief(
        brief_id=brief_id,
        concept_name=f"concept-{brief_id}",
        concept_description="Workshop visual concept",
        images=("Rotating magnets on a pin",),
        diagram_summary="Field lines between poles",
        explanations=("Motion from magnetic interaction.",),
        experiment_prompt="Observe tabletop magnet motion.",
        questions=("What did the demo show?",),
        evidence_confidence=confidence,
        teacher_notes="Internal only",
    )


def test_visual_approval_store_persists_decisions() -> None:
    store = SQLiteRecordStore(":memory:")
    approval_store = VisualApprovalStore(store)
    brief = _brief("dvis_a")
    approval = approve_visual_brief(brief, approved=True, approved_by="teacher-manual")
    approval_store.save(approval)

    loaded = approval_store.get("dvis_a")
    assert loaded is not None
    assert loaded.approved
    assert loaded.approved_by == "teacher-manual"


def test_pending_briefs_excludes_decided() -> None:
    store = SQLiteRecordStore(":memory:")
    workflow = VisualApprovalWorkflow(store)
    briefs = (_brief("dvis_a"), _brief("dvis_b"))
    workflow.record(approve_visual_brief(briefs[0], approved=True))

    pending = workflow.pending_briefs(briefs)
    assert len(pending) == 1
    assert pending[0].brief_id == "dvis_b"


def test_selective_resolve_exports_only_approved() -> None:
    store = SQLiteRecordStore(":memory:")
    workflow = VisualApprovalWorkflow(store)
    high = _brief("dvis_high", confidence=0.95)
    low = _brief("dvis_low", confidence=0.4)
    briefs = (high, low)

    workflow.record(approve_visual_brief(high, approved=True, max_images=1))
    workflow.record(
        approve_visual_brief(
            low,
            approved=False,
            approved_by="teacher-rejected",
            review_note="Low confidence",
        )
    )

    approvals = workflow.resolve(briefs, auto_approve=False)
    exports = export_approved_briefs(briefs, approvals, curriculum_topic="kids-plasma")

    assert len(exports) == 1
    assert exports[0].concept_name == "concept-dvis_high"


def test_resolve_visual_approvals_auto_persists() -> None:
    store = SQLiteRecordStore(":memory:")
    brief = _brief("dvis_auto")
    approvals = resolve_visual_approvals(
        (brief,),
        store=store,
        auto_approve=True,
        min_confidence=0.7,
        persist=True,
    )
    assert len(approvals) == 1
    assert approvals[0].approved

    stored = VisualApprovalStore(store).get("dvis_auto")
    assert stored is not None
    assert stored.approved
