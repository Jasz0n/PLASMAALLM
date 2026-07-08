"""Tests for KEL-steered loop policy."""

from allm.kel.layer import KnowledgeEvaluationLayer
from allm.kel.types import KELConfig
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import KelSteeringConfig, KelSteeringPolicy, LoopConfig, StudentIteration
from allm.loop.learning_loop import IterationReport
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState


def _kel_with_metrics(**metrics: float) -> KnowledgeEvaluationLayer:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="topic-a", description="A"))
    state = KnowledgeState(store)
    kel = KnowledgeEvaluationLayer(graph, store, state, KELConfig(high_gst=0.5))
    for name, value in metrics.items():
        store.put("kel_metrics", name, {"value": value}, reason="test")
    store.put(
        "kel_snapshots",
        "graph",
        {"nodes": ["topic-a"], "edges": [], "taken_at": "2026-01-01T00:00:00Z"},
        reason="test",
    )
    return kel


def _report(score: float) -> IterationReport:
    return IterationReport(
        iteration=1,
        students=(
            StudentIteration(
                student_id="s1",
                score_before=0.0,
                score_after=score,
                goals=("topic-a",),
                samples_studied=4,
            ),
        ),
        debate_disagreement=None,
        compression_applied=0,
        compression_retracted=0,
    )


def test_mastery_raises_exam_size() -> None:
    kel = _kel_with_metrics(rcr=0.3, cd=0.1, gst=0.5, crr=2.0, lg=0.1, cre=0.8, ghs=0.5)
    policy = KelSteeringPolicy(KelSteeringConfig(mastery_threshold=0.7))
    active = LoopConfig(questions_per_exam=4, samples_per_iteration=8)
    decision = policy.decide(2, [_report(0.85)], kel, active)
    assert decision.questions_per_exam == 6
    assert not decision.halt


def test_static_illusion_halts_loop() -> None:
    kel = _kel_with_metrics(rcr=0.3, cd=0.1, gst=0.95, crr=0.2, lg=0.1, cre=0.2, ghs=0.4)
    store = kel._store
    for value in (0.3, 0.2, 0.1, 0.0, -0.1):
        store.put("kel_metrics", "lg", {"value": value}, reason="lg")
    policy = KelSteeringPolicy(
        KelSteeringConfig(min_iterations_before_halt=6, min_lg_history_for_halt=5)
    )
    decision = policy.decide(6, [_report(0.0)] * 5, kel, LoopConfig())
    assert decision.halt
    assert decision.reason


def test_static_illusion_does_not_halt_early() -> None:
    kel = _kel_with_metrics(rcr=0.3, cd=0.1, gst=0.95, crr=0.2, lg=0.1, cre=0.2, ghs=0.4)
    kel._store.put("kel_metrics", "lg", {"value": -0.1}, reason="lg")
    policy = KelSteeringPolicy(KelSteeringConfig(min_iterations_before_halt=6))
    decision = policy.decide(3, [_report(0.1), _report(0.0)], kel, LoopConfig())
    assert not decision.halt


def test_static_illusion_skipped_when_peak_improving() -> None:
    kel = _kel_with_metrics(rcr=0.3, cd=0.1, gst=0.95, crr=0.2, lg=0.1, cre=0.2, ghs=0.4)
    for value in (0.3, 0.2, 0.1, 0.0, -0.1):
        kel._store.put("kel_metrics", "lg", {"value": value}, reason="lg")
    policy = KelSteeringPolicy(
        KelSteeringConfig(
            min_iterations_before_halt=6,
            min_lg_history_for_halt=5,
            strategy_advance_threshold=0.35,
        )
    )
    decision = policy.decide(6, [_report(0.3)] * 5, kel, LoopConfig())
    assert not decision.halt


def test_stagnation_boosts_samples() -> None:
    kel = _kel_with_metrics(rcr=0.3, cd=0.1, gst=0.5, crr=2.0, lg=0.0, cre=0.8, ghs=0.5)
    policy = KelSteeringPolicy()
    active = LoopConfig(samples_per_iteration=16)
    decision = policy.decide(
        3,
        [_report(0.0), _report(0.0)],
        kel,
        active,
    )
    assert decision.samples_per_iteration == 24
