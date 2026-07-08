"""Tests for M43 multi-objective KEL and benchmark suite."""

from __future__ import annotations

from allm.benchmarks.suite import BenchmarkSuite
from allm.evaluator.independent import IndependentEvaluator
from allm.evaluator.types import EvaluationInput
from allm.kel.objectives import compromise_decision, compromise_score, ObjectiveWeights


def test_compromise_score_weighted() -> None:
    from allm.evaluator.independent import EvaluationSnapshot

    snapshot = EvaluationSnapshot(
        learning=0.8,
        stability=0.3,
        retention=0.4,
        generalization=0.5,
        evidence_quality=0.6,
        review_efficiency=0.7,
        contradiction_health=0.5,
    )
    score = compromise_score(
        snapshot,
        ObjectiveWeights(learning=0.1, stability=0.5, retention=0.4),
    )
    assert score < 0.55


def test_compromise_maintain_when_unstable() -> None:
    inputs = EvaluationInput(
        student_id="s1",
        kel_ks=0.35,
        kel_lg=0.1,
        heldout_first=0.3,
        heldout_last=0.25,
        heldout_peak=0.4,
    )
    snapshot = IndependentEvaluator().evaluate(inputs)
    decision = compromise_decision(snapshot)
    assert decision.mode in {"maintain", "repair"}


def test_compromise_learn_when_healthy() -> None:
    inputs = EvaluationInput(
        student_id="s1",
        kel_ks=0.82,
        kel_lg=0.15,
        heldout_first=0.3,
        heldout_last=0.55,
        heldout_peak=0.55,
    )
    snapshot = IndependentEvaluator().evaluate(inputs)
    decision = compromise_decision(snapshot)
    assert decision.mode == "learn"


def test_benchmark_suite_five_dimensions() -> None:
    inputs = EvaluationInput(
        student_id="s1",
        kel_ks=0.5,
        kel_lg=-0.1,
        heldout_first=0.4,
        heldout_last=0.3,
        heldout_peak=0.45,
        kel_cd=0.15,
        kel_cre=0.6,
    )
    results = BenchmarkSuite().evaluate(inputs)
    assert len(results) == 5
    assert {row.dimension.value for row in results} == {
        "retention",
        "generalization",
        "adaptation",
        "scientific_reasoning",
        "engineering",
    }


def test_benchmark_net_improvement() -> None:
    suite = BenchmarkSuite()
    inputs = EvaluationInput(student_id="s1", kel_ks=0.4, heldout_last=0.3, heldout_peak=0.4)
    before = suite.evaluate(inputs)
    after = suite.evaluate(
        EvaluationInput(student_id="s1", kel_ks=0.6, heldout_last=0.35, heldout_peak=0.4)
    )
    deltas = suite.net_improvement(before, after)
    assert deltas["retention"] > 0
