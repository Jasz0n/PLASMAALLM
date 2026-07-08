"""Tests for M41 dependency-weighted risk, retrieval strength, book review budget."""

from __future__ import annotations

from allm.collector.pool import SamplePool
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop.learning_loop import LoopConfig
from allm.loop.maintenance_curriculum import collect_curriculum_mix, MaintenanceSplit
from allm.models import EchoModel, ModelSpec
from allm.planner.dependency_risk import dependency_boost
from allm.planner.forgetting_risk import topic_forgetting_risk
from allm.planner.retrieval_strength import retrieval_strength
from allm.students import ModelStudent
from allm.students.failures import FailureLog
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig


def test_dependency_boost_rises_with_dependents() -> None:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="linear-algebra", description="base"))
    graph.add(
        Concept(
            name="machine-learning",
            description="applied",
            prerequisites=("linear-algebra",),
        )
    )
    assert dependency_boost(graph, "linear-algebra") > dependency_boost(graph, "machine-learning")


def test_retrieval_strength_from_exams() -> None:
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    samples = [
        Sample(id="s0", input="plasma?", target="gas", metadata={"topic": "plasma"}),
    ]
    teacher = Teacher(
        state,
        DatasetExamGenerator(samples),
        ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=1.0),
    )
    student = ModelStudent(
        "s1",
        "plasma",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    student.study("plasma?", "gas")
    teacher.evaluate(student, teacher.create_exam(topics=["plasma"], num_questions=1, seed=1))
    assert retrieval_strength(state, "s1", "plasma") == 1.0
    student.replace_notes({})
    teacher.evaluate(student, teacher.create_exam(topics=["plasma"], num_questions=1, seed=2))
    strength = retrieval_strength(state, "s1", "plasma")
    assert strength is not None and strength < 1.0


def test_book_phase_reserves_review_slots() -> None:
    pool = SamplePool()
    pool.ingest(
        [
            Sample(
                id="r1",
                input="review?",
                target="old",
                metadata={"topic": "kids-plasma", "sample_kind": "definition"},
            ),
            Sample(
                id="r2",
                input="review2?",
                target="old2",
                metadata={"topic": "kids-plasma", "sample_kind": "definition"},
            ),
        ]
    )
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    primary = [
        Sample(
            id=f"b{i}",
            input=f"What is term{i}?",
            target=f"def{i}",
            metadata={"topic": "kids-plasma", "pin": True},
        )
        for i in range(32)
    ]
    collected, counts = collect_curriculum_mix(
        pool=pool,
        failures=FailureLog(store),
        state=state,
        student_id="s1",
        goal_topics=["kids-plasma"],
        maintenance_topics=[],
        cfg=LoopConfig(samples_per_iteration=32),
        split=MaintenanceSplit(0.7, 0.2, 0.1),
        primary_samples=primary,
        planner_review_topics=["kids-plasma"],
    )
    assert counts["review"] > 0
    assert counts["primary"] <= 23
    assert len(collected) <= 32


def test_forgetting_risk_includes_dependency() -> None:
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="base", description="b"))
    graph.add(Concept(name="child", description="c", prerequisites=("base",)))
    base = topic_forgetting_risk(state, "s1", "base", graph=graph)
    child = topic_forgetting_risk(state, "s1", "child", graph=graph)
    assert base >= child
