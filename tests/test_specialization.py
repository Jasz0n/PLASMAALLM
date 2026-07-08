"""Tests for student identity, mission routing, and expert lookup."""

from pathlib import Path

import pytest

from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.knowledge import Concept, KnowledgeGraph
from allm.planner import IngestRouter, NeedPlanner, TopicInfo, build_signals
from allm.students import ScriptedStudent, StudentIdentity, domain_fit, load_identity
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, best_expert, rank_experts

ROOT = Path(__file__).resolve().parents[1]


def test_domain_fit_primary_and_ignored() -> None:
    identity = StudentIdentity(
        student_id="plasma-student",
        primary_domains=("plasma", "matter"),
        ignored_domains=("history",),
        exploration_rate=0.0,
    )
    fit, reason = domain_fit("kids-plasma", identity)
    assert fit == 1.0
    assert reason == "primary mission"

    fit, reason = domain_fit("medieval-history", identity)
    assert fit == 0.0
    assert reason == "ignored domain"


def test_load_plasma_identity_merges_shared_core() -> None:
    identity = load_identity(ROOT / "configs/students/plasma_student.yaml")
    assert identity.student_id == "plasma-student"
    assert "scientific-reasoning" in identity.core_domains
    assert "plasma" in identity.primary_domains


def test_mission_weights_zero_outside_mission() -> None:
    identity = StudentIdentity(
        student_id="software-student",
        primary_domains=("python", "rust"),
        exploration_rate=0.0,
    )
    catalog = {
        "python-basics": TopicInfo(importance=0.9),
        "kids-plasma": TopicInfo(importance=0.9),
    }
    store = SQLiteRecordStore(":memory:")
    signals = build_signals(KnowledgeState(store), "software-student", catalog, identity=identity)
    by_topic = {row.topic: row for row in signals}
    assert by_topic["python-basics"].importance == pytest.approx(0.9)
    assert by_topic["kids-plasma"].importance == 0.0


def test_ingest_router_assigns_specialists() -> None:
    plasma = StudentIdentity(
        student_id="plasma-student",
        primary_domains=("plasma", "matter"),
        ignored_domains=("history", "medieval"),
        exploration_rate=0.0,
    )
    software = StudentIdentity(
        student_id="software-student",
        primary_domains=("fastify", "python"),
        ignored_domains=("plasma", "medieval"),
        exploration_rate=0.0,
    )
    router = IngestRouter([plasma, software], seed=1)
    routing = router.route_document(["kids-plasma", "fastify-api", "medieval-history"])
    assert "plasma-student" in routing["kids-plasma"]
    assert "software-student" in routing["fastify-api"]
    assert "medieval-history" not in routing


def test_expert_ranking(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "s.sqlite3")
    state = KnowledgeState(store)
    samples = [
        __import__("allm.data.base", fromlist=["Sample"]).Sample(
            id="s1", input="What is plasma?", target="fields", metadata={"topic": "plasma"}
        ),
    ]
    teacher = Teacher(state, DatasetExamGenerator(samples), ExactMatchGrader("contains"))
    teacher.evaluate(
        ScriptedStudent("plasma-student", "plasma", knowledge={"What is plasma?": "fields"}),
        teacher.create_exam(num_questions=1, seed=1),
    )
    teacher.evaluate(
        ScriptedStudent("software-student", "plasma", knowledge={"What is plasma?": "wrong"}),
        teacher.create_exam(num_questions=1, seed=2),
    )
    ranking = rank_experts(state, "plasma")
    assert ranking.rankings[0][0] == "plasma-student"
    assert best_expert(state, "plasma") == "plasma-student"
    store.close()


def test_mission_planner_prefers_specialty_topics(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "s.sqlite3")
    state = KnowledgeState(store)
    identity = load_identity(ROOT / "configs/students/plasma_student.yaml")
    catalog = {
        "kids-plasma": TopicInfo(importance=0.9),
        "fastify-api": TopicInfo(importance=0.9),
    }
    signals = build_signals(state, identity.student_id, catalog, identity=identity)
    roadmap = NeedPlanner().plan(identity.student_id, signals)
    goals = [item.topic for item in roadmap.items if item.need > 0]
    assert goals[0] == "kids-plasma"
    assert "fastify-api" not in goals
    store.close()
