"""Outbound webhook dispatch (M51 slice 3): opt-in, approval-gated, audited."""

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from allm.events import (
    ApprovalError,
    EventLog,
    WebhookDispatcher,
    WebhookRegistry,
)
from allm.storage import SQLiteRecordStore


class RecordingSender:
    """A test transport that captures calls instead of hitting the network."""

    def __init__(self, status: int = 200) -> None:
        self.calls: list[tuple[str, bytes, dict]] = []
        self._status = status

    def send(self, url: str, body: bytes, headers: dict) -> tuple[int, str]:
        self.calls.append((url, body, headers))
        return self._status, f"{self._status} OK"


@pytest.fixture()
def wired(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "wh.sqlite3")
    registry = WebhookRegistry(store)
    sender = RecordingSender()
    dispatcher = WebhookDispatcher(registry, store, sender=sender)
    log = EventLog(store, on_emit=dispatcher.dispatch)
    return registry, dispatcher, log, sender


def test_nothing_delivers_until_a_human_approves(wired) -> None:
    registry, _dispatcher, log, sender = wired
    sub = registry.register("https://p.example/hook", event_types=("confidence.changed",))
    assert sub.status == "proposed"

    log.emit("confidence.changed", "plasma", {"value": 0.7})
    assert sender.calls == []  # opt-in: a proposed subscription is silent

    registry.approve(sub.id, approver="jasz0n", reason="vetted platform endpoint")
    log.emit("proposal.opened", "prop_1", {})  # not subscribed
    log.emit("confidence.changed", "plasma", {"value": 0.8})  # subscribed
    assert len(sender.calls) == 1
    assert sender.calls[0][2]["X-ALLM-Event"] == "confidence.changed"


def test_approval_requires_a_named_human(wired) -> None:
    registry, *_ = wired
    sub = registry.register("https://p.example/hook")
    with pytest.raises(ApprovalError, match="named human"):
        registry.approve(sub.id, approver="   ", reason="x")


def test_payload_is_hmac_signed_with_the_subscription_secret(wired) -> None:
    registry, _dispatcher, log, sender = wired
    sub = registry.register("https://p.example/hook", secret="k" * 16)
    registry.approve(sub.id, approver="ada", reason="ok")
    log.emit("confidence.changed", "plasma", {"value": 0.9})

    _url, body, headers = sender.calls[0]
    expected = "sha256=" + hmac.new(b"k" * 16, body, hashlib.sha256).hexdigest()
    assert headers["X-ALLM-Signature"] == expected
    payload = json.loads(body)
    assert payload["event"]["type"] == "confidence.changed"
    assert "wire_version" in payload


def test_disabled_subscription_stops_and_deliveries_are_recorded(wired) -> None:
    registry, dispatcher, log, sender = wired
    sub = registry.register("https://p.example/hook")
    registry.approve(sub.id, approver="ada", reason="ok")
    log.emit("proposal.opened", "prop_1", {})
    registry.disable(sub.id, reason="rotating endpoint")
    log.emit("proposal.opened", "prop_2", {})

    assert len(sender.calls) == 1  # only the pre-disable event
    deliveries = dispatcher.recent_deliveries()
    assert len(deliveries) == 1 and deliveries[0].ok and deliveries[0].status_code == 200


def test_a_failing_endpoint_never_breaks_the_core_write(tmp_path: Path) -> None:
    class Boom:
        def send(self, url, body, headers):
            raise ConnectionError("refused")

    store = SQLiteRecordStore(tmp_path / "boom.sqlite3")
    registry = WebhookRegistry(store)
    dispatcher = WebhookDispatcher(registry, store, sender=Boom())
    log = EventLog(store, on_emit=dispatcher.dispatch)
    sub = registry.register("https://down.example/hook")
    registry.approve(sub.id, approver="ada", reason="ok")

    event = log.emit("confidence.changed", "plasma", {"value": 0.5})  # must not raise
    assert event.seq == 1  # the event was still written
    failed = dispatcher.recent_deliveries()[0]
    assert failed.ok is False and "ConnectionError" in failed.detail


def test_registry_rejects_non_http_urls(wired) -> None:
    registry, *_ = wired
    with pytest.raises(ValueError, match="http"):
        registry.register("ftp://host/path")


def test_api_webhook_lifecycle_never_leaks_the_secret(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_app
    from allm.api.security import AllowAllVerifier

    sender = RecordingSender()
    app = create_app(
        tmp_path / "api.sqlite3", verifier=AllowAllVerifier(), webhook_sender=sender
    )
    with TestClient(app) as client:
        reg = client.post(
            "/webhooks",
            json={"url": "https://p.example/hook", "event_types": ["confidence.changed"]},
        ).json()
        assert reg["status"] == "proposed" and "secret" in reg  # returned once

        # before approval: no delivery
        client.post("/evidence", json={"claim": "c", "concept": "plasma",
                                       "contributor": "ada", "outcome": "supported"})
        assert sender.calls == []

        # listing never carries the secret
        assert all("secret" not in s for s in client.get("/webhooks").json())

        approved = client.post(
            f"/webhooks/{reg['id']}/approve",
            json={"approver": "jasz0n", "reason": "vetted"},
        ).json()
        assert approved["approved_by"] == "jasz0n" and approved["status"] == "approved"

        client.post("/evidence", json={"claim": "c2", "concept": "plasma",
                                       "contributor": "bob", "outcome": "supported"})
        assert [c[2]["X-ALLM-Event"] for c in sender.calls] == ["confidence.changed"]

        deliveries = client.get("/webhooks/deliveries").json()
        assert deliveries and deliveries[0]["ok"] and deliveries[0]["status_code"] == 200

        assert client.post("/webhooks", json={"url": "ftp://x/y"}).status_code == 422
        assert client.post(
            "/webhooks/wh_missing/approve", json={"approver": "x", "reason": "y"}
        ).status_code == 404
