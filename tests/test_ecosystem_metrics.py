"""Tests for Researcher ecosystem metrics and KEL feed."""

from allm.kel import KELConfig, KnowledgeEvaluationLayer
from allm.knowledge import Concept, KnowledgeGraph
from allm.researcher.ecosystem_metrics import ResearcherEcosystemMetrics, compute_ecosystem_metrics
from allm.researcher.types import KnowledgePackage, PackageConcept, ResearchRecommendation
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState


def test_ecosystem_metrics_missing_knowledge() -> None:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="kids-plasma", description="plasma"))
    state = KnowledgeState(store)
    recommendations = [
        ResearchRecommendation(
            topic="fastify-api",
            priority=0.8,
            reason="test",
            package_id="kpkg_x",
            provider="test",
        )
    ]
    packages = [
        KnowledgePackage.build(
            provider="test",
            title="t",
            concepts=(PackageConcept(name="fastify-api", confidence=0.7),),
        )
    ]
    metrics = compute_ecosystem_metrics(graph, state, recommendations, packages, store=store)
    assert metrics.missing_knowledge == 1.0
    assert metrics.emerging_topics == 1


def test_kel_diagnose_research_gap() -> None:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    state = KnowledgeState(store)
    kel = KnowledgeEvaluationLayer(
        graph,
        store,
        state,
        config=KELConfig(high_missing_knowledge=0.3),
    )
    kel.evaluate(
        ecosystem=ResearcherEcosystemMetrics(
            missing_knowledge=0.9,
            emerging_topics=5,
            recommendation_count=10,
        )
    )
    modes = {finding.mode for finding in kel.diagnose()}
    assert "research_gap" in modes
