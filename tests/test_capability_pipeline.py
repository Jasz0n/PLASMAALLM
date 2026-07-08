"""Tests for capability-driven Researcher pipeline."""

from pathlib import Path

from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.capabilities.planning import ResearchPlan
from allm.researcher.capabilities.registry import DEFAULT_PIPELINE, get_capability
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_default_pipeline_has_all_levels() -> None:
    names = set(DEFAULT_PIPELINE)
    assert "observe.curiosity" in names
    assert "analysis.gap" in names
    assert "missions.review" in names
    assert "discovery.video" in names
    assert "understanding.sync" in names
    assert "understanding.vision" in names
    assert "understanding.audio" in names
    assert "understanding.ocr" in names
    assert "discovery.livekit" in names
    assert "understanding.livestream" in names
    assert "understanding.livekit.archive" in names
    assert "planning.research" in names
    assert "discovery.workshop" in names
    assert "discovery.book" in names
    assert "verification.cross_source" in names
    assert "understanding.book.images" in names
    assert "understanding.package" in names
    assert "verification.graph" in names
    assert "curriculum.target" in names
    assert "ecosystem.analyze" in names
    assert "economy.ledger" in names
    assert "improvement.reflect" in names


def test_researcher_cycle_via_pipeline() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
        workshop_max_files=0,
        catalog_topics=(DEFAULT_TOPIC, "fastify-api"),
    )
    report = researcher.run_cycle()
    assert report.packages
    assert report.recommendations
    assert report.capability_summary
    assert len(report.capability_summary) >= len(DEFAULT_PIPELINE)
    assert researcher.active_recommendations()[0].priority > 0


def test_workshop_aligned_topics() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        workshop_max_files=1,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()
    topics = {rec.topic for rec in report.recommendations}
    assert DEFAULT_TOPIC in topics


def test_plan_capability_produces_steps() -> None:
    cap = get_capability("planning.research")
    from allm.researcher.capabilities.base import CapabilityContext, PipelineState
    from allm.researcher.capabilities.base import ResearcherPipelineConfig

    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
        ),
    )
    result = cap.run(ctx, PipelineState())
    plan = result.artifacts["plan"]
    assert isinstance(plan, ResearchPlan)
    assert plan.steps
