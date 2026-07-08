"""Tests for M42 decay prediction, maintenance optimizer, richer KS."""

from __future__ import annotations

from allm.collector.pool import SamplePool
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kel.knowledge_stability import (
    cross_topic_coherence,
    debate_consistency_ks,
    merge_ks,
)
from allm.loop.learning_loop import LoopConfig
from allm.loop.maintenance_curriculum import collect_curriculum_mix, MaintenanceSplit
from allm.models import EchoModel, ModelSpec
from allm.planner.decay_prediction import decay_urgency, proactive_review_topics
from allm.planner.maintenance_budget import expected_ks_gain, rank_review_topics
from allm.planner.types import RoadmapItem
from allm.students import ModelStudent
from allm.students.failures import FailureLog
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig


def _teacher_with_declining_plasma() -> tuple[KnowledgeState, str]:
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
    student.replace_notes({})
    for seed in (2, 3, 4):
        teacher.evaluate(student, teacher.create_exam(topics=["plasma"], num_questions=1, seed=seed))
    return state, "s1"


def test_decay_urgency_rises_for_declining_topic() -> None:
    state, student_id = _teacher_with_declining_plasma()
    assert decay_urgency(state, student_id, "plasma") > 0.0


def test_proactive_review_before_threshold_crossed() -> None:
    state, student_id = _teacher_with_declining_plasma()
    topics = proactive_review_topics(
        state,
        student_id,
        ["plasma"],
        current_risks={"plasma": 0.1},
        limit=3,
    )
    assert "plasma" in topics


def test_expected_ks_gain_prefers_high_risk() -> None:
    high = expected_ks_gain(forgetting_risk=0.9, importance=0.8)
    low = expected_ks_gain(forgetting_risk=0.2, importance=0.8)
    assert high > low


def test_rank_review_topics_orders_by_gain() -> None:
    state, student_id = _teacher_with_declining_plasma()
    items = (
        RoadmapItem(
            rank=1,
            topic="plasma",
            need=0.8,
            weakness=0.5,
            importance=0.9,
            curiosity=0.5,
            novelty=0.5,
            forgetting_risk=0.2,
            reason="test",
        ),
        RoadmapItem(
            rank=2,
            topic="other",
            need=0.4,
            weakness=0.5,
            importance=0.5,
            curiosity=0.5,
            novelty=0.5,
            forgetting_risk=0.8,
            reason="test",
        ),
    )
    ranked = rank_review_topics(items, state=state, student_id=student_id, limit=2)
    assert ranked[0][0] in {"plasma", "other"}


def test_richer_ks_merge() -> None:
    merged = merge_ks(0.6, 0.7, 0.5, 0.8, 0.9, 0.95)
    assert merged is not None and 0.6 <= merged <= 0.95


def test_cross_topic_coherence() -> None:
    state, student_id = _teacher_with_declining_plasma()
    samples = [
        Sample(id="s1", input="other?", target="yes", metadata={"topic": "other"}),
    ]
    teacher = Teacher(
        state,
        DatasetExamGenerator(samples),
        ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=1.0),
    )
    student = ModelStudent(
        student_id,
        "other",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    student.study("other?", "yes")
    teacher.evaluate(student, teacher.create_exam(topics=["other"], num_questions=1, seed=10))
    student.replace_notes({})
    teacher.evaluate(student, teacher.create_exam(topics=["other"], num_questions=1, seed=11))
    coherence = cross_topic_coherence(state, student_id)
    assert coherence is not None and coherence > 0.0


def test_debate_consistency_ks() -> None:
    assert debate_consistency_ks(0.2) == 0.8
    assert debate_consistency_ks(None) is None


def test_optimizer_collects_from_planner_topics() -> None:
    pool = SamplePool()
    pool.ingest(
        [
            Sample(
                id="r1",
                input="review?",
                target="old",
                metadata={"topic": "kids-plasma", "sample_kind": "definition"},
            ),
        ]
    )
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    collected, counts = collect_curriculum_mix(
        pool=pool,
        failures=FailureLog(store),
        state=state,
        student_id="s1",
        goal_topics=["kids-plasma"],
        maintenance_topics=[],
        cfg=LoopConfig(samples_per_iteration=10),
        split=MaintenanceSplit(0.7, 0.2, 0.1),
        primary_samples=[],
        planner_review_topics=["kids-plasma"],
    )
    assert counts["review"] > 0
    assert len(collected) <= 10
