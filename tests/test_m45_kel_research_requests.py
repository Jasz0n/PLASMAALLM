"""Tests for M45 KEL research requests."""

from __future__ import annotations

from allm.kel.research_requests import (
    KelResearchRequest,
    build_kel_research_requests,
)
from allm.kel.types import Finding
from allm.loop.learning_loop import IterationReport, LoopConfig, StudentIteration
from allm.loop.retention_gates import RetentionContext
from allm.researcher.layer import ResearcherLayer
from allm.researcher.remediation import requests_to_recommendations
from allm.storage import SQLiteRecordStore
from allm.trainer.forgetting import ForgettingReport


def _report(iteration: int, score: float, strategy: str = "definitions") -> IterationReport:
    return IterationReport(
        iteration=iteration,
        students=(
            StudentIteration(
                student_id="s1",
                score_before=0.0,
                score_after=score,
                goals=("kids-plasma",),
                samples_studied=16,
                strategy=strategy,
            ),
        ),
        debate_disagreement=None,
        compression_applied=0,
        compression_retracted=0,
    )


def test_build_research_request_on_repair_mode() -> None:
    requests = build_kel_research_requests(
        findings=(),
        compromise_mode="repair",
        retention=None,
        reports=[_report(1, 0.2)],
        student_id="s1",
        topics=("kids-plasma",),
        strategy="definitions",
        kel_ks=0.26,
    )
    assert requests
    assert any(row.trigger == "repair_mode" for row in requests)


def test_build_strategy_stagnation_request() -> None:
    reports = [_report(i, 0.2) for i in range(1, 5)]
    requests = build_kel_research_requests(
        findings=(),
        compromise_mode="maintain",
        retention=None,
        reports=reports,
        student_id="s1",
        topics=("kids-plasma",),
        strategy="definitions",
        kel_ks=0.3,
    )
    assert any(row.trigger == "strategy_stagnation" for row in requests)


def test_requests_to_remediation_recommendations() -> None:
    request = KelResearchRequest(
        id="r1",
        topic="kids-plasma",
        task="Find visual explanations",
        trigger="repair_mode",
        priority=0.9,
        student_id="s1",
        search_hints=("visual",),
        reason="KS low",
    )
    recs = requests_to_recommendations([request])
    assert recs[0].recommendation_kind == "remediation"
    assert recs[0].provider == "kel-research"
    assert "visual" in (recs[0].proposal_hint or "")


def test_researcher_submits_kel_requests() -> None:
    store = SQLiteRecordStore(":memory:")
    layer = ResearcherLayer(store)
    request = KelResearchRequest(
        id="r1",
        topic="kids-plasma",
        task="Find diagrams",
        trigger="unstable_mastery",
        priority=0.85,
        student_id="kids-kel",
        reason="unstable",
    )
    count = layer.submit_kel_research_requests((request,))
    assert count == 1
    assert len(layer.kel_research_requests()) == 1
    active = layer.active_recommendations()
    assert active[0].recommendation_kind == "remediation"


def test_unstable_mastery_finding_generates_request() -> None:
    findings = (
        Finding(mode="unstable_mastery", detail="KS 0.26 below 0.70"),
    )
    requests = build_kel_research_requests(
        findings=findings,
        compromise_mode="repair",
        retention=None,
        reports=[_report(1, 0.1)],
        student_id="s1",
        topics=("kids-plasma",),
        strategy="definitions",
        kel_ks=0.26,
    )
    triggers = {row.trigger for row in requests}
    assert "unstable_mastery" in triggers
