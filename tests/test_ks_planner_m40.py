"""Tests for M40 KS-driven planner."""

from __future__ import annotations

import pytest

from allm.collector.pool import SamplePool
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.loop.maintenance_curriculum import adaptive_maintenance_split
from allm.models import EchoModel, ModelSpec
from allm.planner import NeedPlanner, TopicInfo, build_signals
from allm.planner.forgetting_risk import topic_forgetting_risk
from allm.students import ModelStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig


def test_adaptive_maintenance_split_low_ks() -> None:
    split = adaptive_maintenance_split(0.3)
    assert split.review_fraction == 0.40
    assert split.new_fraction == 0.4


def test_adaptive_maintenance_split_high_ks() -> None:
    split = adaptive_maintenance_split(0.9)
    assert split.new_fraction == 0.9
    assert split.review_fraction == 0.05


def test_topic_forgetting_risk_from_maintenance_list() -> None:
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    risk = topic_forgetting_risk(
        state,
        "s1",
        "kids-plasma",
        maintenance_topics={"kids-plasma"},
        global_ks=0.5,
    )
    assert risk >= 0.9


def test_planner_boosts_at_risk_topics(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_KS_PLANNER", "1")
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    catalog = {
        "kids-plasma": TopicInfo(importance=0.8, curiosity=0.5),
        "other": TopicInfo(importance=0.8, curiosity=0.5),
    }
    samples = [
        Sample(id="s0", input="q?", target="a", metadata={"topic": "kids-plasma"}),
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
    student.study("q?", "a")
    teacher.evaluate(student, teacher.create_exam(topics=["kids-plasma"], num_questions=1, seed=1))
    student.replace_notes({})
    teacher.evaluate(student, teacher.create_exam(topics=["kids-plasma"], num_questions=1, seed=2))

    signals = build_signals(
        state,
        "s1",
        catalog,
        maintenance_topics=("kids-plasma",),
        global_ks=0.4,
    )
    roadmap = NeedPlanner().plan("s1", signals)
    plasma = next(item for item in roadmap.items if item.topic == "kids-plasma")
    other = next(item for item in roadmap.items if item.topic == "other")
    assert plasma.forgetting_risk >= other.forgetting_risk
    assert "forgetting" in plasma.reason
