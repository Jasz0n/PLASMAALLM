"""Tests for allm.storage — the never-overwrite guarantee above all."""

from pathlib import Path

import pytest

from allm.storage import SQLiteRecordStore, storage_backends


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteRecordStore:
    s = SQLiteRecordStore(tmp_path / "test.sqlite3")
    yield s
    s.close()


def test_put_and_get_latest(store: SQLiteRecordStore) -> None:
    store.put("beliefs", "gravity", {"confidence": 0.4})
    store.put("beliefs", "gravity", {"confidence": 0.8}, reason="passed exam")
    latest = store.get("beliefs", "gravity")
    assert latest is not None
    assert latest.version == 2
    assert latest.value == {"confidence": 0.8}
    assert latest.reason == "passed exam"


def test_history_preserves_all_versions(store: SQLiteRecordStore) -> None:
    for confidence in (0.1, 0.5, 0.9):
        store.put("beliefs", "gravity", {"confidence": confidence})
    history = store.history("beliefs", "gravity")
    assert [r.version for r in history] == [1, 2, 3]
    assert [r.value["confidence"] for r in history] == [0.1, 0.5, 0.9]


def test_missing_key_returns_none(store: SQLiteRecordStore) -> None:
    assert store.get("beliefs", "nothing") is None
    assert store.history("beliefs", "nothing") == []


def test_namespaces_are_isolated(store: SQLiteRecordStore) -> None:
    store.put("beliefs", "k", {"v": 1})
    store.put("failures", "k", {"v": 2})
    assert store.get("beliefs", "k").value == {"v": 1}
    assert store.keys("beliefs") == ["k"]
    assert store.keys("failures") == ["k"]


def test_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persist.sqlite3"
    first = SQLiteRecordStore(path)
    first.put("ns", "k", {"v": 1})
    first.close()
    second = SQLiteRecordStore(path)
    assert second.get("ns", "k").value == {"v": 1}
    second.close()


def test_registered_as_backend() -> None:
    assert storage_backends.get("sqlite") is SQLiteRecordStore
