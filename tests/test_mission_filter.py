"""Tests for mission-based sample filtering in the learning loop."""

from allm.data.base import Sample
from allm.loop.learning_loop import LearningLoop
from allm.students.identity import StudentIdentity


def test_filter_by_mission_keeps_primary_topics() -> None:
    identity = StudentIdentity(
        student_id="plasma-student",
        primary_domains=("plasma",),
        exploration_rate=0.0,
    )
    samples = [
        Sample(id="a", input="q1", target="a1", metadata={"topic": "kids-plasma"}),
        Sample(id="b", input="q2", target="a2", metadata={"topic": "fastify-api"}),
    ]
    kept = LearningLoop._filter_by_mission(samples, identity, mission_seed=1)
    assert len(kept) == 1
    assert kept[0].id == "a"
