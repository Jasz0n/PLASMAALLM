"""The official Python client (M52 integration kit), against the real app."""

from pathlib import Path

import pytest

from allm.client import AllmClient, AllmError, Response

TOKEN = "k" * 20


class _TestClientTransport:
    """Drives the in-process ASGI app — no network, real request path."""

    def __init__(self, app) -> None:
        from fastapi.testclient import TestClient

        self._tc = TestClient(app)

    def request(self, method, url, headers, body) -> Response:
        r = self._tc.request(method, url, headers=headers, content=body)
        return Response(r.status_code, r.content, dict(r.headers))


@pytest.fixture()
def client(tmp_path: Path) -> AllmClient:
    pytest.importorskip("fastapi")
    from allm.api.app import create_app
    from allm.api.security import StaticTokenVerifier

    app = create_app(tmp_path / "api.sqlite3", verifier=StaticTokenVerifier(TOKEN))
    return AllmClient(base_url="http://testserver", token=TOKEN, transport=_TestClientTransport(app))


def test_client_drives_the_whole_contributor_loop(client: AllmClient) -> None:
    assert client.health()["status"] == "ok"
    assert client.ready()["status"] == "ready"
    assert client.wire()["wire_version"]  # the contract is reachable

    ingested = client.submit_documents(
        [{"name": "intro", "text": "A plasma is an ionized gas; we call it the fourth state of matter."}]
    )
    assert ingested["units"] >= 1

    result = client.submit_evidence(
        claim="plasma conducts", concept="plasma", contributor="ada", outcome="supported"
    )
    assert result["package_id"].startswith("pkg_")
    assert 0.0 <= result["confidence"]["value"] <= 1.0

    concept = client.concept("plasma")
    assert "provenance" in concept and concept["concept"]["name"] == "plasma"

    # the feed reflects the write, read back through the client
    feed = client.catch_up()
    assert [e["type"] for e in feed[-2:]] == ["evidence.submitted", "confidence.changed"]


def test_errors_surface_as_typed_exceptions(client: AllmClient) -> None:
    with pytest.raises(AllmError) as unknown:
        client.concept("does-not-exist")
    assert unknown.value.status == 404 and unknown.value.type == "http_error"

    # a schema failure carries the per-field detail through
    with pytest.raises(AllmError) as invalid:
        client.submit_evidence(claim="x", concept="y", contributor="z", outcome="not-a-valid-outcome")
    assert invalid.value.status == 422 and invalid.value.type == "validation_error"
    assert invalid.value.fields


def test_writes_need_a_token_before_a_request_is_sent(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from allm.api.app import create_app
    from allm.api.security import StaticTokenVerifier

    app = create_app(tmp_path / "api.sqlite3", verifier=StaticTokenVerifier(TOKEN))
    anon = AllmClient(base_url="http://testserver", transport=_TestClientTransport(app))

    # reads work without a token
    assert anon.concepts() == []
    # writes fail fast, client-side, with a clear message
    with pytest.raises(AllmError, match="needs a token"):
        anon.submit_evidence(claim="a", concept="b", contributor="c", outcome="supported")
