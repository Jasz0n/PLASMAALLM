"""API hardening: auth, rate limits, size caps, pagination (M50)."""

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from allm.api.app import create_app  # noqa: E402
from allm.api.security import (  # noqa: E402
    AllowAllVerifier,
    RateLimiter,
    StaticTokenVerifier,
)

TOKEN = "s3cret-token-of-16+"


def submission(**extra) -> dict:
    return {
        "claim": "converter exceeds 80% efficiency",
        "concept": "Energy Converter",
        "contributor": "alice",
        "outcome": "supported",
        **extra,
    }


@pytest.fixture()
def secured(tmp_path: Path):
    app = create_app(
        tmp_path / "api.sqlite3", verifier=StaticTokenVerifier(TOKEN)
    )
    with TestClient(app) as client:
        yield client


def test_writes_require_the_token(secured: TestClient) -> None:
    assert secured.post("/evidence", json=submission()).status_code == 401
    wrong = secured.post(
        "/evidence", json=submission(), headers={"Authorization": "Bearer nope-nope-nope-nope"}
    )
    assert wrong.status_code == 401
    right = secured.post(
        "/evidence", json=submission(), headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert right.status_code == 201
    assert secured.post("/kel/evaluate").status_code == 401


def test_reads_stay_open(secured: TestClient) -> None:
    assert secured.get("/health").status_code == 200
    assert secured.get("/concepts").status_code == 200
    assert secured.get("/proposals").status_code == 200


def test_short_tokens_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="16 characters"):
        StaticTokenVerifier("short")


def test_rate_limit_returns_429(tmp_path: Path) -> None:
    app = create_app(
        tmp_path / "api.sqlite3",
        verifier=AllowAllVerifier(),
        rate_limiter=RateLimiter(requests=3, per_seconds=3600),
    )
    with TestClient(app) as client:
        codes = [
            client.post("/evidence", json=submission(claim=f"claim {i}")).status_code
            for i in range(5)
        ]
    assert codes[:3] == [201, 201, 201]
    assert codes[3:] == [429, 429]


def test_rate_limiter_refills_over_time() -> None:
    limiter = RateLimiter(requests=2, per_seconds=10)
    assert limiter.allow("k", now=0.0)
    assert limiter.allow("k", now=0.0)
    assert not limiter.allow("k", now=0.0)
    assert limiter.allow("k", now=5.0)  # half the window refills one token


def test_size_caps_reject_abuse(tmp_path: Path) -> None:
    app = create_app(tmp_path / "api.sqlite3", verifier=AllowAllVerifier())
    with TestClient(app) as client:
        huge_claim = client.post("/evidence", json=submission(claim="x" * 5_000))
        assert huge_claim.status_code == 422
        huge_doc = client.post(
            "/documents", json=[{"name": "war-and-peace", "text": "x" * 600_000}]
        )
        assert huge_doc.status_code == 422
        too_many = client.post(
            "/documents",
            json=[{"name": f"d{i}", "text": "plasma is energy"} for i in range(51)],
        )
        assert too_many.status_code == 413


def test_concepts_paginate(tmp_path: Path) -> None:
    app = create_app(tmp_path / "api.sqlite3", verifier=AllowAllVerifier())
    with TestClient(app) as client:
        for i in range(7):
            client.post("/evidence", json=submission(concept=f"Concept {i}"))
        first = client.get("/concepts", params={"limit": 3}).json()
        second = client.get("/concepts", params={"limit": 3, "offset": 3}).json()
        assert len(first) == 3 and len(second) == 3
        assert {c["name"] for c in first}.isdisjoint({c["name"] for c in second})
        assert client.get("/concepts", params={"limit": 0}).status_code == 422


def test_env_default_uses_static_token(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALLM_API_TOKEN", TOKEN)
    app = create_app(tmp_path / "api.sqlite3")
    with TestClient(app) as client:
        assert client.post("/kel/evaluate").status_code == 401
        ok = client.post(
            "/kel/evaluate", headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert ok.status_code == 200


def test_published_openapi_contract_is_current(tmp_path: Path) -> None:
    """docs/openapi.json is the frozen wire contract (M50) — regenerating
    must produce the same paths and version, or the publication is stale."""
    import json

    from allm.api.app import create_app

    published = json.loads(
        (Path(__file__).resolve().parents[1] / "docs" / "openapi.json").read_text()
    )
    live = create_app(tmp_path / "contract.sqlite3").openapi()
    assert published["info"]["version"] == live["info"]["version"]
    assert sorted(published["paths"]) == sorted(live["paths"])
