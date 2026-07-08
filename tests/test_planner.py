"""Tests for allm.planner: need scoring, dependencies, signal assembly."""

from pathlib import Path

import pytest

from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.planner import (
    NeedPlanner,
    NeedPlannerConfig,
    Planner,
    TopicInfo,
    TopicSignal,
    build_signals,
    load_catalog,
    planners,
)
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher


def signal(topic: str, **overrides) -> TopicSignal:
    return TopicSignal(topic=topic, **overrides)


def test_registered_and_protocol() -> None:
    assert planners.get("need") is NeedPlanner
    assert isinstance(NeedPlanner(), Planner)


def test_signal_derived_factors() -> None:
    s = signal("t", confidence=0.75, observations=1)
    assert s.weakness == 0.25
    assert s.novelty == 0.5
    assert signal("t").weakness == 1.0  # unexamined = fully weak
    assert signal("t").novelty == 1.0


def test_need_ordering_weakest_first() -> None:
    roadmap = NeedPlanner().plan(
        "s1",
        [
            signal("strong", confidence=0.9, observations=2),
            signal("weak", confidence=0.1, observations=2),
        ],
    )
    assert [i.topic for i in roadmap.items] == ["weak", "strong"]
    assert roadmap.items[0].rank == 1
    assert roadmap.items[0].need > roadmap.items[1].need


def test_importance_and_curiosity_scale_need() -> None:
    roadmap = NeedPlanner().plan(
        "s1",
        [
            signal("dull", confidence=0.2, importance=0.1, curiosity=0.1),
            signal("exciting", confidence=0.2, importance=0.9, curiosity=0.9),
        ],
    )
    assert roadmap.items[0].topic == "exciting"


def test_blocked_topics_sort_last() -> None:
    roadmap = NeedPlanner().plan(
        "s1",
        [
            signal("quantum-gravity", dependencies=("general-relativity",)),
            signal("general-relativity", confidence=0.2, observations=1),
            signal("arithmetic", confidence=0.4, observations=3),
        ],
    )
    assert roadmap.items[-1].topic == "quantum-gravity"
    assert roadmap.items[-1].blocked_by == ("general-relativity",)
    assert "blocked by" in roadmap.items[-1].reason


def test_blocked_urgency_boosts_prerequisite() -> None:
    config = NeedPlannerConfig(blocked_boost=0.9)
    # prerequisite alone would score lower than the blocked topic
    roadmap = NeedPlanner(config).plan(
        "s1",
        [
            signal("advanced", importance=1.0, curiosity=1.0, dependencies=("basics",)),
            signal("basics", confidence=0.4, observations=9, importance=0.1, curiosity=0.1),
        ],
    )
    basics = next(i for i in roadmap.items if i.topic == "basics")
    advanced = next(i for i in roadmap.items if i.topic == "advanced")
    assert basics.need == pytest.approx(0.9 * advanced.need)
    assert basics.rank < advanced.rank


def test_unknown_dependency_counts_as_unmet() -> None:
    roadmap = NeedPlanner().plan("s1", [signal("t", dependencies=("never-seen",))])
    assert roadmap.items[0].blocked_by == ("never-seen",)


def test_mastered_dependency_unblocks() -> None:
    roadmap = NeedPlanner().plan(
        "s1",
        [
            signal("advanced", dependencies=("basics",)),
            signal("basics", confidence=0.8, observations=2),
        ],
    )
    advanced = next(i for i in roadmap.items if i.topic == "advanced")
    assert advanced.blocked_by == ()
    assert advanced.rank == 1  # weakest unblocked topic leads


def test_to_goals_skips_zero_need() -> None:
    roadmap = NeedPlanner().plan(
        "s1",
        [signal("mastered", confidence=1.0, observations=3), signal("weak", confidence=0.2)],
    )
    goals = roadmap.to_goals()
    assert [g.topic for g in goals] == ["weak"]


def test_roadmap_to_goals() -> None:
    roadmap = NeedPlanner().plan(
        "s1", [signal("a", confidence=0.1), signal("b", confidence=0.9)]
    )
    goals = roadmap.to_goals(max_goals=1)
    assert len(goals) == 1
    assert goals[0].topic == "a"
    assert goals[0].student_id == "s1"
    assert 0.0 <= goals[0].priority <= 1.0


def test_build_signals_merges_catalog_and_state(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "s.sqlite3")
    state = KnowledgeState(store)
    samples = [
        Sample(id="s1", input="2+2?", target="4", metadata={"topic": "math"}),
        Sample(id="s2", input="3*3?", target="9", metadata={"topic": "math"}),
    ]
    teacher = Teacher(state, DatasetExamGenerator(samples), ExactMatchGrader())
    teacher.evaluate(
        ScriptedStudent("kid", "math", knowledge={"2+2?": "4", "3*3?": "9"}),
        teacher.create_exam(num_questions=2, seed=1),
    )

    catalog = {
        "math": TopicInfo(importance=0.9),
        "physics": TopicInfo(curiosity=0.8, dependencies=("math",)),
    }
    signals = build_signals(state, "kid", catalog)
    by_topic = {s.topic: s for s in signals}
    assert by_topic["math"].confidence == 1.0
    assert by_topic["math"].observations == 1
    assert by_topic["math"].importance == 0.9
    assert by_topic["physics"].confidence is None
    assert by_topic["physics"].dependencies == ("math",)
    store.close()


def test_load_catalog(tmp_path: Path) -> None:
    file = tmp_path / "catalog.yaml"
    file.write_text(
        "math:\n  importance: 0.9\nphysics:\n  dependencies: [math]\nchemistry:\n",
        encoding="utf-8",
    )
    catalog = load_catalog(file)
    assert catalog["math"].importance == 0.9
    assert catalog["physics"].dependencies == ("math",)
    assert catalog["chemistry"].importance == 0.5  # defaults
