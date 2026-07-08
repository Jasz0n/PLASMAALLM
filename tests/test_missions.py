"""Tests for Researcher missions."""

from allm.researcher.missions import MissionStore, ResearchMission
from allm.storage import SQLiteRecordStore


def test_mission_build_has_default_tasks() -> None:
    mission = ResearchMission.build(goal="Understand plasma", target_topics=("kids-plasma",))
    assert mission.id.startswith("rmission_")
    assert mission.status == "open"
    assert mission.tasks


def test_open_from_gap_is_idempotent() -> None:
    store = SQLiteRecordStore(":memory:")
    missions = MissionStore(store)
    first = missions.open_from_gap(parent="plasma", child="fusion", missing="ions")
    second = missions.open_from_gap(parent="plasma", child="fusion", missing="ions")
    assert first.id == second.id
    assert len(missions.active()) == 1


def test_active_missions_sorted_by_priority() -> None:
    store = SQLiteRecordStore(":memory:")
    missions = MissionStore(store)
    low = ResearchMission.build(goal="low", priority=0.3)
    high = ResearchMission.build(goal="high", priority=0.9)
    missions.save(low)
    missions.save(high)
    active = missions.active()
    assert active[0].goal == "high"
