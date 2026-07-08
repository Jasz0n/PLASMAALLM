"""Tests for Researcher layer."""

from pathlib import Path

from allm.planner.researcher_signals import merge_research_recommendations
from allm.planner.signals import TopicInfo
from allm.researcher import ResearcherLayer
from allm.researcher.packages import package_from_samples_jsonl
from allm.researcher.types import ResearchRecommendation
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_package_from_software_fixture() -> None:
    path = ROOT / "transcripts/Software/samples_dev.jsonl"
    package = package_from_samples_jsonl(path, provider="software-fixture", title="dev")
    assert package.concepts
    assert package.definitions
    assert package.provider == "software-fixture"


def test_researcher_cycle_enqueues_recommendations() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
        workshop_max_files=0,
    )
    report = researcher.run_cycle()
    assert report.packages
    assert report.recommendations
    active = researcher.active_recommendations()
    assert active[0].priority > 0


def test_merge_research_recommendations() -> None:
    catalog = {"kids-plasma": TopicInfo(importance=0.5, curiosity=0.5)}
    recs = [
        ResearchRecommendation(
            topic="kids-plasma",
            priority=0.9,
            reason="test",
            package_id="kpkg_x",
            provider="test",
        )
    ]
    merged = merge_research_recommendations(catalog, recs)
    assert merged["kids-plasma"].importance > 0.5
