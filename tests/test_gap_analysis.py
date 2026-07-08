"""Tests for graph gap analysis and knowledge tiers."""

from allm.knowledge import Concept, KnowledgeGraph
from allm.researcher.capabilities.base import CapabilityContext, PipelineState, ResearcherPipelineConfig
from allm.researcher.capabilities.gap_analysis import GraphGapAnalysisCapability, analyze_graph_gaps
from allm.researcher.knowledge_tier import classify_knowledge_tier
from allm.researcher.missions import MissionStore
from allm.researcher.types import KnowledgePackage, PackageConcept
from allm.researcher.capabilities.verification import _verify_package
from allm.storage import SQLiteRecordStore


def test_analyze_graph_gaps_finds_missing_prerequisite() -> None:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="fusion", prerequisites=("ions",)))
    report = analyze_graph_gaps(graph)
    assert report.gaps
    assert report.gaps[0].missing_prerequisite == "ions"


def test_gap_capability_opens_missions() -> None:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="fusion", prerequisites=("ions",)))
    ctx = CapabilityContext(store=store, config=ResearcherPipelineConfig(), graph=graph)
    cap = GraphGapAnalysisCapability()
    pipeline = PipelineState()
    cap.run(ctx, pipeline)
    missions = MissionStore(store).active()
    assert missions
    assert "ions" in missions[0].target_topics


def test_classify_knowledge_tier() -> None:
    assert classify_knowledge_tier(
        in_graph=True, graph_confidence=0.9, has_conflict=False, package_confidence=0.8
    ) == "established"
    assert classify_knowledge_tier(
        in_graph=False, graph_confidence=None, has_conflict=False, package_confidence=0.6
    ) == "emerging"
    assert classify_knowledge_tier(
        in_graph=True, graph_confidence=0.9, has_conflict=True, package_confidence=0.8
    ) == "hypothesis"


def test_verify_package_assigns_tiers() -> None:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="kids-plasma", confidence=0.9))
    package = KnowledgePackage.build(
        provider="test",
        title="test",
        concepts=(PackageConcept(name="kids-plasma", confidence=0.8), PackageConcept(name="novel", confidence=0.6)),
    )
    verified, _ = _verify_package(package, graph, reputation=0.7)
    tiers = {concept.name: concept.knowledge_tier for concept in verified.concepts}
    assert tiers["kids-plasma"] == "established"
    assert tiers["novel"] == "emerging"
