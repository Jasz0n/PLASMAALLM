"""Tests for M38 retention-gated KEL steering."""

from __future__ import annotations

from allm.data.base import Sample
from allm.kel.layer import KnowledgeEvaluationLayer
from allm.kel.types import KELConfig
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop.kel_steering import KelSteeringConfig, KelSteeringPolicy
from allm.loop.learning_loop import IterationReport, LoopConfig, StudentIteration
from allm.loop.retention_gates import (
    HeldoutRetentionTracker,
    build_retention_context,
    reset_strategy_for_new_phase,
)
from allm.loop.strategy import LearningStrategy
from allm.researcher.types import KnowledgePackage, PackageConcept
from allm.storage import SQLiteRecordStore
from allm.students.model_student import ModelStudent, ModelStudentConfig
from allm.teacher import KnowledgeState
from allm.teacher.source_training import BOOK_PROVIDER, filter_workshop_delta_samples
from allm.trainer.forgetting import ForgettingReport


def _kel() -> KnowledgeEvaluationLayer:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="topic-a", description="A"))
    return KnowledgeEvaluationLayer(graph, store, KnowledgeState(store), KELConfig(high_gst=0.5))


def _report(score: float, *, iteration: int = 1) -> IterationReport:
    return IterationReport(
        iteration=iteration,
        students=(
            StudentIteration(
                student_id="s1",
                score_before=0.0,
                score_after=score,
                goals=("topic-a",),
                samples_studied=4,
                strategy="definitions",
            ),
        ),
        debate_disagreement=None,
        compression_applied=0,
        compression_retracted=0,
    )


def test_retention_gate_blocks_strategy_advance() -> None:
    policy = KelSteeringPolicy(
        KelSteeringConfig(
            strategy_advance_threshold=0.35,
            require_retention_stable=True,
            retention_max_drop_from_peak=0.10,
        )
    )
    tracker = HeldoutRetentionTracker()
    tracker.record(0.36, phase="book")
    reports = [_report(0.36), _report(0.10, iteration=2)]
    retention = build_retention_context(
        reports,
        tracker,
        current_phase="workshop",
        max_drop_from_peak=0.10,
        require_stable=True,
    )
    assert not retention.retention_stable
    active = LoopConfig(strategy="definitions")
    decision = policy.decide(3, reports, _kel(), active, retention=retention)
    assert decision.strategy is None


def test_retention_allows_advance_when_stable() -> None:
    policy = KelSteeringPolicy(
        KelSteeringConfig(strategy_advance_threshold=0.35, require_retention_stable=True)
    )
    tracker = HeldoutRetentionTracker()
    reports = [_report(0.12), _report(0.23), _report(0.36, iteration=3)]
    tracker.record(0.36, phase="book")
    retention = build_retention_context(
        reports,
        tracker,
        current_phase="book",
        max_drop_from_peak=0.15,
        require_stable=True,
    )
    assert retention.retention_stable
    decision = policy.decide(4, reports, _kel(), LoopConfig(), retention=retention)
    assert decision.strategy == LearningStrategy.RELATIONS


def test_forgetting_blocks_strategy_advance() -> None:
    policy = KelSteeringPolicy(
        KelSteeringConfig(
            strategy_advance_threshold=0.35,
            block_advance_on_forgetting=True,
        )
    )
    reports = [
        IterationReport(
            iteration=1,
            students=_report(0.40).students,
            debate_disagreement=None,
            compression_applied=0,
            compression_retracted=0,
            forgetting=(
                ForgettingReport(
                    student_id="s1",
                    probed_topics=("kids-plasma",),
                    regressions={"kids-plasma": -0.20},
                ),
            ),
        )
    ]
    retention = build_retention_context(
        reports,
        HeldoutRetentionTracker(),
        current_phase="book",
        max_drop_from_peak=0.15,
        require_stable=True,
    )
    decision = policy.decide(2, reports, _kel(), LoopConfig(), retention=retention)
    assert decision.strategy is None


def test_reset_strategy_for_new_phase() -> None:
    active = LoopConfig(
        strategy="research",
        sample_kinds=("definition", "we_call", "compact", "teaching"),
        samples_per_iteration=72,
        use_exam_paraphrase=True,
    )
    reset = reset_strategy_for_new_phase(active)
    assert reset.strategy == "definitions"
    assert reset.sample_kinds == ("definition", "we_call")
    assert reset.use_exam_paraphrase is False


def test_pinned_book_notes_survive_eviction() -> None:
    from allm.models.base import ModelSpec
    from allm.models.echo import EchoModel

    student = ModelStudent(
        "kid",
        "kids-plasma",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
        ModelStudentConfig(max_notes=3, notes_in_prompt=3),
    )
    student.study("What is plasma?", "ionized gas", pinned=True)
    student.study("What is field?", "magnetic influence")
    student.study("What is matter?", "substance")
    student.study("What is energy?", "capacity to do work")
    assert student.pinned_note_count() == 1
    assert ("What is plasma?", "ionized gas") in student.notes


def test_workshop_delta_prefers_workshop_only_concepts() -> None:
    workshop = KnowledgePackage(
        id="ws1",
        provider="kids-workshops",
        title="Workshop",
        concepts=(
            PackageConcept(name="Magnetic Field Demo", description=""),
            PackageConcept(name="Unique Workshop Topic", description=""),
        ),
    )
    book = KnowledgePackage(
        id="b1",
        provider=BOOK_PROVIDER,
        title="Book",
        concepts=(PackageConcept(name="Magnetic Field", description=""),),
    )
    samples = [
        Sample(id="s1", input="magnetic?", target="field", metadata={}),
        Sample(id="s2", input="unique?", target="workshop topic", metadata={}),
    ]
    filtered = filter_workshop_delta_samples(samples, (workshop, book))
    assert any(row.id == "s2" for row in filtered)
