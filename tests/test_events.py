"""Domain event stream — the platform's live feed (M51), offline."""

from pathlib import Path

import pytest

from allm.events import Event, EventLog
from allm.storage import SQLiteRecordStore


@pytest.fixture()
def log(tmp_path: Path) -> EventLog:
    return EventLog(SQLiteRecordStore(tmp_path / "events.sqlite3"))


def test_events_are_ordered_and_pollable_by_cursor(log: EventLog) -> None:
    a = log.emit("proposal.opened", "prop_1", {"origin": "conflict"})
    b = log.emit("confidence.changed", "plasma", {"value": 0.7})
    c = log.emit("proposal.resolved", "prop_1", {"status": "resolved"})
    assert [a.seq, b.seq, c.seq] == [1, 2, 3]

    # a fresh subscriber sees everything, oldest first
    assert [e.seq for e in log.since(0)] == [1, 2, 3]
    # advancing the cursor yields only the un-seen tail — no replay
    tail = log.since(cursor=b.seq)
    assert [e.type for e in tail] == ["proposal.resolved"]
    assert log.since(cursor=c.seq) == []  # caught up

    assert log.count() == 3
    assert [e.seq for e in log.latest(limit=2)] == [3, 2]  # newest first


def test_events_survive_reload(tmp_path: Path) -> None:
    path = tmp_path / "persist.sqlite3"
    log = EventLog(SQLiteRecordStore(path))
    log.emit("evidence.submitted", "pkg_1", {"concept": "X"})
    # a new EventLog over the same store keeps numbering monotonically
    reopened = EventLog(SQLiteRecordStore(path))
    second = reopened.emit("evidence.submitted", "pkg_2", {"concept": "Y"})
    assert second.seq == 2
    assert [e.subject for e in reopened.since(0)] == ["pkg_1", "pkg_2"]
    assert isinstance(reopened.since(0)[0], Event)


def test_api_emits_and_serves_events(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_app
    from allm.api.security import AllowAllVerifier

    app = create_app(tmp_path / "api.sqlite3", verifier=AllowAllVerifier())
    with TestClient(app) as client:
        client.post(
            "/evidence",
            json={"claim": "c", "concept": "plasma", "contributor": "ada",
                  "outcome": "supported"},
        )
        feed = client.get("/events").json()
        types = [e["type"] for e in feed["events"]]
        # one submission emits both the fact and the belief change
        assert types == ["evidence.submitted", "confidence.changed"]
        assert feed["cursor"] == 2 and feed["total"] == 2
        conf = next(e for e in feed["events"] if e["type"] == "confidence.changed")
        assert conf["subject"] == "plasma" and "value" in conf["data"]

        # polling from the cursor returns nothing new
        assert client.get("/events", params={"since": feed["cursor"]}).json()["events"] == []

        # the dashboard mirrors the feed on its read side
        state = client.get("/dashboard/state").json()
        assert state["events"]["total"] == 2
