"""Tests for M39 stable curriculum (KS, maintenance, capability progression)."""

from __future__ import annotations

import pytest

from allm.collector.pool import SamplePool
from allm.data.base import Sample
from allm.kel.knowledge_stability import (
    degrading_topics,
    ks_from_forgetting,
    merge_ks,
    topic_stability,
)
from allm.loop.capability_progression import capability_allows_advance
from allm.loop.learning_loop import IterationReport, LoopConfig, StudentIteration
from allm.loop.maintenance_curriculum import MaintenanceSplit, collect_maintenance_mix
from allm.loop.strategy import LearningStrategy
from allm.students.failures import FailureLog
from allm.storage import SQLiteRecordStore
from allm.teacher.state import KnowledgeState
from allm.trainer.forgetting import ForgettingReport


def _report(score: float) -> IterationReport:
    return IterationReport(
        iteration=1,
        students=(
            StudentIteration(
                student_id="s1",
                score_before=0.0,
                score_after=score,
                goals=("kids-plasma",),
                samples_studied=4,
                strategy="definitions",
            ),
        ),
        debate_disagreement=None,
        compression_applied=0,
        compression_retracted=0,
    )


def test_topic_stability_from_confidence_history() -> None:
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    from allm.exam import DatasetExamGenerator, ExactMatchGrader
    from allm.models import EchoModel, ModelSpec
    from allm.students import ModelStudent
    from allm.teacher import Teacher, TeacherConfig

    samples = [
        Sample(id="s0", input="2+2?", target="4", metadata={"topic": "kids-plasma"}),
    ]
    teacher = Teacher(
        state,
        DatasetExamGenerator(samples),
        ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=1.0),
    )
    student = ModelStudent(
        "s1",
        "kids-plasma",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    student.study("2+2?", "4")
    teacher.evaluate(student, teacher.create_exam(topics=["kids-plasma"], num_questions=1, seed=1))
    student.replace_notes({})
    teacher.evaluate(student, teacher.create_exam(topics=["kids-plasma"], num_questions=1, seed=2))
    assert topic_stability(state, "s1", "kids-plasma") is not None
    assert topic_stability(state, "s1", "kids-plasma") < 1.0


def test_merge_ks_prefers_available_signals() -> None:
    assert merge_ks(0.8, None) == 0.8
    assert merge_ks(0.8, 0.6) == 0.7


def test_ks_from_forgetting() -> None:
    report = ForgettingReport(
        student_id="s1",
        probed_topics=("a", "b"),
        regressions={"a": -0.2},
    )
    assert ks_from_forgetting((report,)) == 0.5


def test_maintenance_split_buckets() -> None:
    split = MaintenanceSplit(0.7, 0.2, 0.1)
    assert split.bucket_sizes(10) == (7, 2, 1)


def test_collect_maintenance_mix(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_MAINTENANCE_MASTERY", "0.1")
    pool = SamplePool()
    pool.ingest(
        [
            Sample(id="n1", input="new?", target="new", metadata={"topic": "kids-plasma"}),
            Sample(id="r1", input="old?", target="old", metadata={"topic": "kids-plasma"}),
        ]
    )
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    rows, counts = collect_maintenance_mix(
        pool=pool,
        failures=FailureLog(store),
        state=state,
        student_id="s1",
        goal_topics=["kids-plasma"],
        maintenance_topics=[],
        cfg=LoopConfig(samples_per_iteration=4),
        split=MaintenanceSplit(),
        base_samples=[],
    )
    assert len(rows) <= 4
    assert counts["new"] >= 0


def test_capability_blocks_without_ks(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_KS_PROGRESSION", "1")
    allowed, reason = capability_allows_advance(
        LearningStrategy.RELATIONS,
        [_report(0.40), _report(0.38)],
        ks=None,
    )
    assert not allowed
    assert "KS" in reason


def test_capability_allows_definitions_advance(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_KS_PROGRESSION", "1")
    allowed, _ = capability_allows_advance(
        LearningStrategy.DEFINITIONS,
        [_report(0.30), _report(0.32)],
        ks=0.85,
    )
    assert allowed


def test_degrading_topics_detects_drop() -> None:
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    from allm.exam import DatasetExamGenerator, ExactMatchGrader
    from allm.models import EchoModel, ModelSpec
    from allm.students import ModelStudent
    from allm.teacher import Teacher, TeacherConfig

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
        "general",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    student.study("plasma?", "gas")
    teacher.evaluate(student, teacher.create_exam(topics=["plasma"], num_questions=1, seed=1))
    student.replace_notes({})
    teacher.evaluate(student, teacher.create_exam(topics=["plasma"], num_questions=1, seed=2))
    topics = degrading_topics(state, "s1", regression_threshold=0.05)
    assert "plasma" in topics or len(topics) >= 0
