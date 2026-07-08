"""Tests for visual knowledge package distillation."""

from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.visual_distillation import VisualDistillationCapability
from allm.researcher.motion_continuity import enrich_synced_evidence_continuity
from allm.researcher.motion_tracking import StubMotionTracker, enrich_synced_evidence_motion
from allm.researcher.multimodal import discover_video_fixtures, sync_fixtures_with_workshop_dir
from allm.researcher.object_identity import enrich_object_identities
from allm.researcher.packages import package_from_workshop_dir
from allm.researcher.visual_distillation import (
    attach_distilled_visuals,
    distill_visual_evidence,
)
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def _enriched_rows():
    fixtures = discover_video_fixtures(ROOT / "transcripts/Kids/visual")
    synced = sync_fixtures_with_workshop_dir(fixtures, ROOT / "transcripts/Kids/cleaned/mk")
    visual_rows = [row for row in synced if row.visual is not None]
    motion_rows = [
        enrich_synced_evidence_motion(row, tracker=StubMotionTracker())
        for row in visual_rows
    ]
    continuity_rows, _tracks = enrich_synced_evidence_continuity(motion_rows, min_score=0.35)
    enriched, _identities = enrich_object_identities(continuity_rows, min_score=0.30)
    return enriched


def test_distill_visual_evidence_produces_briefs() -> None:
    rows = _enriched_rows()
    briefs = distill_visual_evidence(rows, max_images=3, max_questions=5)
    assert briefs
    brief = briefs[0]
    assert brief.concept_name
    assert brief.images
    assert brief.explanations
    assert brief.questions
    assert brief.source_refs
    assert "Teacher" in brief.teacher_notes


def test_distilled_brief_has_no_frame_paths() -> None:
    rows = _enriched_rows()
    briefs = distill_visual_evidence(rows)
    for brief in briefs:
        for image in brief.images:
            assert ".jpg" not in image
            assert ".mp4" not in image


def test_attach_distilled_visuals_updates_package() -> None:
    package = package_from_workshop_dir(
        ROOT / "transcripts/Kids/cleaned/mk",
        max_files=3,
        curriculum_topic="kids-plasma",
    )
    briefs = distill_visual_evidence(_enriched_rows())
    updated = attach_distilled_visuals(package, briefs)
    assert updated.distilled_visual_briefs
    assert len(updated.distilled_visual_briefs) >= 1


def test_visual_distillation_capability_enriches_pipeline() -> None:
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            workshop_curriculum_topic="kids-plasma",
            enable_visual_distillation=True,
        ),
    )
    pipeline = PipelineState()
    pipeline.multimodal_synced = _enriched_rows()
    pipeline.packages.append(
        package_from_workshop_dir(
            ROOT / "transcripts/Kids/cleaned/mk",
            max_files=3,
            curriculum_topic="kids-plasma",
        )
    )
    result = VisualDistillationCapability().run(ctx, pipeline)
    assert result.metrics.yield_count >= 1
    assert pipeline.packages[0].distilled_visual_briefs
