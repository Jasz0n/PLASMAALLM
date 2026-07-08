"""Tests for Teacher-approved student visual export."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.verification import GraphVerificationCapability
from allm.researcher.capabilities.visual_distillation import VisualDistillationCapability
from allm.researcher.capabilities.visual_export import VisualExportCapability
from allm.researcher.multimodal_types import DistilledVisualBrief
from allm.researcher.packages import package_from_workshop_dir
from allm.researcher.student_visual_export import attach_student_visual_packages
from allm.researcher.visual_distillation import attach_distilled_visuals, distill_visual_evidence
from allm.storage import SQLiteRecordStore
from allm.teacher.visual_export import (
    approve_visual_brief,
    auto_approve_briefs,
    export_approved_briefs,
    export_student_visual,
)

ROOT = Path(__file__).resolve().parents[1]


def _sample_brief() -> DistilledVisualBrief:
    return DistilledVisualBrief(
        brief_id="dvis_test",
        concept_name="plasma motion",
        concept_description="Magnet rotation without fuel",
        images=(
            "Rotating magnets on a pin — motion from field interaction",
            "One magnet chasing the other — blue field region visible",
            "Close-up of repulsion between similar poles",
        ),
        diagram_summary="Labels: north pole, south pole, field lines",
        explanations=("The field showed magnetical beat and rotation.",),
        experiment_prompt="Observe tabletop magnet motion without fuel.",
        questions=(
            "What did the workshop show about plasma motion?",
            "How does motion relate to magnetic fields?",
            "Which diagram elements explain the demo?",
            "What happens when similar poles meet?",
        ),
        source_refs=("knowledgeSeekerWorkshop9@712s",),
        evidence_confidence=0.92,
        teacher_notes="Internal Teacher notes — not for students.",
    )


def test_export_student_visual_strips_teacher_fields() -> None:
    brief = _sample_brief()
    approval = approve_visual_brief(brief, max_images=2, max_questions=2)
    exported = export_student_visual(brief, approval, curriculum_topic="kids-plasma")
    assert exported is not None
    assert len(exported.images) == 2
    assert len(exported.questions) == 2
    assert exported.diagram
    assert exported.experiment
    assert "teacher_notes" not in exported.model_dump()
    assert "source_refs" not in exported.model_dump()


def test_auto_approve_respects_confidence_threshold() -> None:
    high = _sample_brief()
    low = high.model_copy(update={"brief_id": "dvis_low", "evidence_confidence": 0.4})
    approvals = auto_approve_briefs((high, low), min_confidence=0.7)
    exports = export_approved_briefs((high, low), approvals)
    assert len(exports) == 1
    assert exports[0].export_id.startswith("svis_")


def test_attach_student_visual_packages_updates_package() -> None:
    package = package_from_workshop_dir(
        ROOT / "transcripts/Kids/cleaned/mk",
        max_files=2,
        curriculum_topic="kids-plasma",
    )
    brief = _sample_brief()
    exports = export_approved_briefs(
        (brief,),
        (approve_visual_brief(brief),),
        curriculum_topic="kids-plasma",
    )
    updated = attach_student_visual_packages(package, exports)
    assert updated.student_visual_packages
    assert updated.student_visual_packages[0].approved_by == "teacher"


def test_visual_export_capability_after_distillation() -> None:
    from allm.researcher.motion_continuity import enrich_synced_evidence_continuity
    from allm.researcher.motion_tracking import StubMotionTracker, enrich_synced_evidence_motion
    from allm.researcher.multimodal import discover_video_fixtures, sync_fixtures_with_workshop_dir
    from allm.researcher.object_identity import enrich_object_identities

    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    synced = sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
    visual_rows = [row for row in synced if row.visual is not None]
    motion_rows = [
        enrich_synced_evidence_motion(row, tracker=StubMotionTracker())
        for row in visual_rows
    ]
    continuity_rows, _tracks = enrich_synced_evidence_continuity(motion_rows, min_score=0.35)
    enriched, _identities = enrich_object_identities(continuity_rows, min_score=0.30)

    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            workshop_curriculum_topic="kids-plasma",
            enable_visual_distillation=True,
            enable_visual_export=True,
            visual_export_auto_approve=True,
            visual_export_min_confidence=0.7,
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = enriched
    package = package_from_workshop_dir(
        ROOT / "transcripts/Kids/cleaned/mk",
        max_files=3,
        curriculum_topic="kids-plasma",
    )
    briefs = distill_visual_evidence(enriched)
    pipeline.packages.append(attach_distilled_visuals(package, briefs))
    VisualDistillationCapability().run(ctx, pipeline)
    GraphVerificationCapability().run(ctx, pipeline)
    result = VisualExportCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    target = pipeline.verified_packages[0]
    assert target.student_visual_packages
