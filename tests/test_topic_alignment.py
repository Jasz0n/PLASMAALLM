"""Tests for Researcher topic alignment with curriculum pools."""

from pathlib import Path

from allm.kdp.corpus import DEFAULT_TOPIC
from allm.planner.researcher_signals import merge_research_recommendations
from allm.planner.signals import TopicInfo
from allm.researcher import ResearcherLayer
from allm.researcher.topic_alignment import align_recommendation_topic
from allm.researcher.types import ResearchRecommendation
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def test_align_noisy_concept_to_curriculum_topic() -> None:
    topic = align_recommendation_topic(
        "The Beauty Of It",
        curriculum_topic=DEFAULT_TOPIC,
        catalog_topics={DEFAULT_TOPIC},
    )
    assert topic == DEFAULT_TOPIC


def test_align_keeps_software_topic_when_in_catalog() -> None:
    topic = align_recommendation_topic(
        "fastify-api",
        curriculum_topic=None,
        catalog_topics={"fastify-api", "kids-plasma"},
    )
    assert topic == "fastify-api"


def test_workshop_researcher_recommends_kids_plasma() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        workshop_max_files=1,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()
    assert report.recommendations
    topics = {rec.topic for rec in report.recommendations}
    assert DEFAULT_TOPIC in topics
    assert "The Beauty Of It" not in topics


def test_merge_boosts_curriculum_topic_not_fragment() -> None:
    catalog = {DEFAULT_TOPIC: TopicInfo(importance=0.3, curiosity=0.3)}
    recs = [
        ResearchRecommendation(
            topic=DEFAULT_TOPIC,
            priority=0.8,
            reason="aligned",
            package_id="kpkg_x",
            provider="kids-workshops",
            concept="C02 Box",
        )
    ]
    merged = merge_research_recommendations(catalog, recs)
    assert merged[DEFAULT_TOPIC].importance > 0.3
