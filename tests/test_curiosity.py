"""Tests for observe.curiosity capability."""

import shutil
import tempfile
from pathlib import Path

from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.curiosity import ObserveCuriosityCapability, build_curiosity_report
from allm.researcher.ecosystem_metrics import ResearcherEcosystemMetrics
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_curiosity_from_high_missing_knowledge() -> None:
    store = SQLiteRecordStore(":memory:")
    ecosystem = ResearcherEcosystemMetrics(missing_knowledge=0.6, high_conflict_areas=0.4)
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(workshop_curriculum_topic="kids-plasma"),
        ecosystem=ecosystem,
    )
    report = build_curiosity_report(ctx)
    sources = {signal.source for signal in report.signals}
    assert "missing_knowledge" in sources
    assert "high_conflict" in sources
    assert report.signals[0].score >= report.signals[-1].score


def test_curiosity_unsynced_video_signal() -> None:
    store = SQLiteRecordStore(":memory:")
    fixture_dir = Path(tempfile.mkdtemp(prefix="allm-curiosity-fixtures-"))
    shutil.copy(
        ROOT / "transcripts/Kids/visual/workshop9_plasma_demo.json",
        fixture_dir / "workshop9_plasma_demo.json",
    )
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            video_fixture_dir=fixture_dir,
            workshop_curriculum_topic="kids-plasma",
        ),
    )
    report = build_curiosity_report(ctx)
    sources = {signal.source for signal in report.signals}
    assert "unsynced_video" in sources


def test_observe_curiosity_capability_populates_pipeline() -> None:
    store = SQLiteRecordStore(":memory:")
    ctx = CapabilityContext(
        store=store,
        config=ResearcherPipelineConfig(
            workshop_dir=None,
            software_samples=None,
        ),
    )
    cap = ObserveCuriosityCapability()
    pipeline = PipelineState()
    result = cap.run(ctx, pipeline)
    assert result.metrics.yield_count >= 0
    assert isinstance(pipeline.curiosity_signals, list)
