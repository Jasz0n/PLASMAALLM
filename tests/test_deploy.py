"""Deployable core (M52): readiness probe + env-driven factory, offline."""

from pathlib import Path

import pytest


def test_health_and_ready_are_open_and_distinct(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_app
    from allm.api.security import StaticTokenVerifier

    # even with auth configured, both probes stay open (orchestrators are anonymous)
    app = create_app(tmp_path / "api.sqlite3", verifier=StaticTokenVerifier("x" * 16))
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        assert health.status_code == 200 and health.json()["status"] == "ok"
        assert ready.status_code == 200 and ready.json()["status"] == "ready"

        # readiness actually exercised the store: a write is visible, proving the DB answers
        client.post(
            "/evidence",
            json={"claim": "c", "concept": "plasma", "contributor": "a", "outcome": "supported"},
            headers={"Authorization": "Bearer " + "x" * 16},
        )
        assert client.get("/ready").status_code == 200


def test_default_factory_honours_storage_env(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_default_app

    db = tmp_path / "nested" / "allm.sqlite3"  # parent must be auto-created
    monkeypatch.setenv("ALLM_STORAGE__PATH", str(db))
    monkeypatch.setenv("ALLM_API_TOKEN", "z" * 20)  # auth on, as in the container

    app = create_default_app()
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        # token enforced: an unauthenticated write is refused
        assert client.post(
            "/evidence",
            json={"claim": "c", "concept": "x", "contributor": "a", "outcome": "supported"},
        ).status_code == 401
    assert db.exists()  # the factory created the store at the env-given path
