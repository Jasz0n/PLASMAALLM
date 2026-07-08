"""Tests for the HTTP API (skipped when the api extras are missing)."""

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from allm.api.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(tmp_path / "api.sqlite3")
    with TestClient(app) as test_client:
        yield test_client


def submission(contributor: str, outcome: str = "supported", **extra) -> dict:
    return {
        "claim": "converter exceeds 80% efficiency",
        "concept": "Energy Converter",
        "contributor": contributor,
        "outcome": outcome,
        **extra,
    }


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_evidence_submission_creates_concept(client: TestClient) -> None:
    response = client.post("/evidence", json=submission("alice"))
    assert response.status_code == 201
    body = response.json()
    assert body["package_id"].startswith("pkg_")
    assert 0.5 < body["confidence"]["value"] < 0.7

    concepts = client.get("/concepts").json()
    assert [c["name"] for c in concepts] == ["Energy Converter"]

    detail = client.get("/concepts/Energy Converter").json()
    assert detail["evidential_confidence"]["contributors"] == 1
    assert "Evidence" in detail["provenance"]


def test_unknown_concept_404(client: TestClient) -> None:
    assert client.get("/concepts/Phlogiston").status_code == 404


def test_documents_open_proposals_for_conflicts(client: TestClient) -> None:
    response = client.post(
        "/documents",
        json=[
            {"name": "w1.md", "text": "The Stack is a converter that exceeds 80 percent efficiency."},
            {"name": "w2.md", "text": "The Stack is a module capped near 10 percent efficiency."},
        ],
    )
    assert response.status_code == 201
    body = response.json()
    assert body["conflicts"] == 1
    assert len(body["proposals_opened"]) == 1

    proposals = client.get("/proposals", params={"status": "open"}).json()
    assert proposals[0]["origin"] == "conflict"


def test_full_proposal_lifecycle_over_http(client: TestClient) -> None:
    client.post(
        "/documents",
        json=[
            {"name": "w1.md", "text": "The Stack is a converter that exceeds 80 percent efficiency."},
            {"name": "w2.md", "text": "The Stack is a module capped near 10 percent efficiency."},
        ],
    )
    proposal_id = client.get("/proposals").json()[0]["id"]

    claim = client.post(
        f"/proposals/{proposal_id}/claim", json={"contributor": "alice-lab"}
    )
    assert claim.json()["status"] == "claimed"

    resolve = client.post(
        f"/proposals/{proposal_id}/resolve",
        json={
            "packages": [
                submission("alice-lab", outcome="challenged", concept="The Stack")
                | {"concept": "The Stack"},
                submission("bob", outcome="challenged") | {"concept": "The Stack"},
            ]
        },
    )
    assert resolve.status_code == 200
    assert resolve.json()["resolution"]["outcome"] == "challenged"

    # double resolution is a conflict, not a crash
    again = client.post(
        f"/proposals/{proposal_id}/resolve",
        json={"packages": [submission("carol") | {"concept": "The Stack"}]},
    )
    assert again.status_code == 409


def test_kel_evaluate_records_measurement(client: TestClient) -> None:
    client.post("/evidence", json=submission("alice"))
    body = client.post("/kel/evaluate").json()
    assert body["report"]["cd"] == 0.0
    assert body["findings"] == []
