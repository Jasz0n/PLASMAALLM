"""Browser-ready boundary (M52): CORS, SSE live feed, error envelope."""

from pathlib import Path

import pytest


@pytest.fixture()
def client_factory(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_app
    from allm.api.security import AllowAllVerifier

    def make(**kwargs):
        app = create_app(
            tmp_path / f"{len(kwargs)}-{id(kwargs)}.sqlite3",
            verifier=AllowAllVerifier(),
            **kwargs,
        )
        return TestClient(app)

    return make


def _seed(client, n: int) -> None:
    for i in range(n):
        client.post(
            "/evidence",
            json={"claim": f"c{i}", "concept": "plasma", "contributor": "ada", "outcome": "supported"},
        )


def test_cors_allows_a_configured_origin_and_is_off_by_default(client_factory) -> None:
    allowed = client_factory(cors_origins=["https://app.plasma.social"])
    pre = allowed.options(
        "/evidence",
        headers={
            "Origin": "https://app.plasma.social",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization",
        },
    )
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == "https://app.plasma.social"

    # default: no origins configured -> no CORS headers, same-origin only
    closed = client_factory()
    pre2 = closed.options(
        "/evidence",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in pre2.headers


def test_errors_have_a_consistent_envelope(client_factory) -> None:
    client = client_factory()
    # an explicit HTTPException
    not_found = client.get("/concepts/nope")
    assert not_found.status_code == 404
    body = not_found.json()
    assert body["error"]["status"] == 404 and body["error"]["type"] == "http_error"
    assert body["detail"]  # kept for FastAPI-convention consumers

    # a schema validation failure is typed distinctly and lists the fields
    invalid = client.post("/evidence", json={"claim": "c"})  # missing required fields
    assert invalid.status_code == 422
    err = invalid.json()["error"]
    assert err["type"] == "validation_error" and err["fields"]


def test_sse_streams_the_backlog_and_resumes_by_cursor(client_factory) -> None:
    client = client_factory()
    _seed(client, 3)  # -> 6 events (submitted + confidence.changed each)

    stream = client.get("/events/stream?live=false")
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")

    frames = [f for f in stream.text.split("\n\n") if f.startswith("id:")]
    assert len(frames) == 6
    first = dict(line.split(": ", 1) for line in frames[0].splitlines())
    assert first["id"] == "1" and first["event"] == "evidence.submitted"

    # resume: only events after the cursor come back
    resumed = client.get("/events/stream?live=false&since=4")
    ids = [
        line.removeprefix("id: ")
        for line in resumed.text.splitlines()
        if line.startswith("id: ")
    ]
    assert ids == ["5", "6"]


def test_sse_resumes_from_last_event_id_header(client_factory) -> None:
    client = client_factory()
    _seed(client, 2)  # 4 events
    resumed = client.get("/events/stream?live=false", headers={"Last-Event-ID": "3"})
    ids = [
        line.removeprefix("id: ")
        for line in resumed.text.splitlines()
        if line.startswith("id: ")
    ]
    assert ids == ["4"]  # the browser's reconnect cursor wins
