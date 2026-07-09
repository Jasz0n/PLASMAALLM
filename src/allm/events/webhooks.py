"""Outbound webhook dispatch: the event stream, delivered (M51 slice 3).

Delivery crosses the network *outward*, so it inherits the core
invariant — **nothing leaves the system without a human approval
record**. A subscription is *proposed* when registered and delivers
nothing until a named human *approves* it; every attempt is recorded in
the append-only store, so what went out (and what failed) is auditable
like everything else.

Deliberately thin and opt-in: no subscription, no traffic. The default
sender is stdlib ``urllib`` with a short timeout; delivery is best-effort
and never raises back into the core write. A production deployment should
move sending to a queue with retries — noted, not pretended.

SSRF note: an approved URL is trusted because a human vetted it. The
registry still rejects non-HTTP(S) schemes; vetting the *host* (no
internal metadata endpoints) is part of what approval means.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.events.log import Event
from allm.storage.base import RecordStore

logger = get_logger("events.webhooks")

SUBSCRIPTIONS = "webhooks"
DELIVERIES = "webhook_deliveries"


class ApprovalError(PermissionError):
    """Raised when delivery is attempted without a human approval record."""


class WebhookSubscription(BaseModel):
    """A platform endpoint that wants the event feed pushed to it."""

    model_config = ConfigDict(frozen=True)

    id: str
    url: str
    event_types: tuple[str, ...] = ()  # empty = every event
    secret: str  # HMAC key; returned once at registration, never listed
    status: str = "proposed"  # proposed | approved | disabled
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: str | None = None
    approved_at: datetime | None = None
    reason: str | None = None

    def public(self) -> dict[str, Any]:
        """Everything except the secret — safe to list."""
        data = self.model_dump(mode="json")
        data.pop("secret")
        return data

    def wants(self, event_type: str) -> bool:
        return self.status == "approved" and (
            not self.event_types or event_type in self.event_types
        )


class WebhookDelivery(BaseModel):
    """One recorded delivery attempt — the outbound audit trail."""

    model_config = ConfigDict(frozen=True)

    id: str
    subscription_id: str
    event_seq: int
    event_type: str
    url: str
    ok: bool
    status_code: int
    detail: str
    attempted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WebhookSender(Protocol):
    """The transport, injected so delivery is testable without a network."""

    def send(self, url: str, body: bytes, headers: dict[str, str]) -> tuple[int, str]:
        """POST ``body``; return ``(status_code, detail)``. May raise."""
        ...


class UrllibSender:
    """Default transport: stdlib ``urllib`` with a hard timeout, no deps."""

    def __init__(self, timeout: float = 5.0) -> None:
        self._timeout = timeout

    def send(self, url: str, body: bytes, headers: dict[str, str]) -> tuple[int, str]:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return response.status, f"{response.status} {response.reason}"
        except urllib.error.HTTPError as exc:  # a real HTTP status, just not 2xx
            return exc.code, f"{exc.code} {exc.reason}"


class WebhookRegistry:
    """Lifecycle of subscriptions: proposed → approved / disabled."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def register(
        self, url: str, *, event_types: tuple[str, ...] = (), secret: str | None = None
    ) -> WebhookSubscription:
        """Register an endpoint. It is ``proposed`` — it delivers nothing
        until approved. Returns the subscription *including* its secret,
        the one time it is ever exposed."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"webhook url must be http(s): {url!r}")
        subscription = WebhookSubscription(
            id="wh_" + secrets.token_hex(6),
            url=url,
            event_types=tuple(event_types),
            secret=secret or secrets.token_hex(16),
        )
        self._put(subscription, "registered (proposed)")
        return subscription

    def approve(self, subscription_id: str, *, approver: str, reason: str) -> WebhookSubscription:
        """The human-approval gate. Without a named approver, no delivery."""
        if not approver.strip():
            raise ApprovalError("webhook approval requires a named human")
        subscription = self._require(subscription_id)
        approved = subscription.model_copy(
            update={
                "status": "approved",
                "approved_by": approver,
                "approved_at": datetime.now(timezone.utc),
                "reason": reason,
            }
        )
        self._put(approved, f"approved by {approver}: {reason[:60]}")
        return approved

    def disable(self, subscription_id: str, *, reason: str) -> WebhookSubscription:
        subscription = self._require(subscription_id)
        disabled = subscription.model_copy(update={"status": "disabled", "reason": reason})
        self._put(disabled, f"disabled: {reason[:60]}")
        return disabled

    def get(self, subscription_id: str) -> WebhookSubscription | None:
        record = self._store.get(SUBSCRIPTIONS, subscription_id)
        return WebhookSubscription.model_validate(record.value) if record else None

    def all(self) -> list[WebhookSubscription]:
        return [
            WebhookSubscription.model_validate(self._store.get(SUBSCRIPTIONS, key).value)
            for key in self._store.keys(SUBSCRIPTIONS)
        ]

    def approved_for(self, event_type: str) -> list[WebhookSubscription]:
        return [s for s in self.all() if s.wants(event_type)]

    def _require(self, subscription_id: str) -> WebhookSubscription:
        subscription = self.get(subscription_id)
        if subscription is None:
            raise KeyError(f"unknown webhook subscription {subscription_id!r}")
        return subscription

    def _put(self, subscription: WebhookSubscription, reason: str) -> None:
        self._store.put(
            SUBSCRIPTIONS,
            subscription.id,
            json.loads(subscription.model_dump_json()),
            reason=reason,
        )


class WebhookDispatcher:
    """Delivers each event to the approved subscriptions that want it."""

    def __init__(
        self,
        registry: WebhookRegistry,
        store: RecordStore,
        *,
        sender: WebhookSender | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._registry = registry
        self._store = store
        self._sender = sender or UrllibSender(timeout)
        self._lock = threading.Lock()

    def dispatch(self, event: Event) -> list[WebhookDelivery]:
        """Deliver ``event`` to every approved matching subscription.

        Never raises: a bad endpoint yields a recorded failed delivery,
        not an exception into the core write path.
        """
        subscriptions = self._registry.approved_for(event.type)
        deliveries: list[WebhookDelivery] = []
        for subscription in subscriptions:
            deliveries.append(self._deliver(subscription, event))
        return deliveries

    def _deliver(self, subscription: WebhookSubscription, event: Event) -> WebhookDelivery:
        from allm.wire import WIRE_VERSION  # lazy: avoids an import cycle

        body = json.dumps(
            {"wire_version": WIRE_VERSION, "event": event.model_dump(mode="json")},
            sort_keys=True,
        ).encode("utf-8")
        signature = hmac.new(
            subscription.secret.encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-ALLM-Event": event.type,
            "X-ALLM-Event-Seq": str(event.seq),
            "X-ALLM-Signature": f"sha256={signature}",
        }
        try:
            status_code, detail = self._sender.send(subscription.url, body, headers)
            ok = 200 <= status_code < 300
        except Exception as exc:  # transport failure, timeout, DNS, ...
            status_code, detail, ok = 0, f"{type(exc).__name__}: {exc}"[:200], False
        return self._record(subscription, event, ok, status_code, detail)

    def _record(
        self,
        subscription: WebhookSubscription,
        event: Event,
        ok: bool,
        status_code: int,
        detail: str,
    ) -> WebhookDelivery:
        with self._lock:
            seq = len(self._store.keys(DELIVERIES)) + 1
            delivery = WebhookDelivery(
                id=f"whd_{seq:012d}",
                subscription_id=subscription.id,
                event_seq=event.seq,
                event_type=event.type,
                url=subscription.url,
                ok=ok,
                status_code=status_code,
                detail=detail,
            )
            self._store.put(
                DELIVERIES,
                f"{seq:012d}",
                json.loads(delivery.model_dump_json()),
                reason=f"{'ok' if ok else 'FAIL'} {event.type} -> {subscription.id}",
            )
        logger.info(
            "webhook %s %s -> %s (%s)",
            "delivered" if ok else "failed", event.type, subscription.id, detail,
        )
        return delivery

    def recent_deliveries(self, *, limit: int = 50) -> list[WebhookDelivery]:
        keys = self._store.keys(DELIVERIES)[-limit:]
        return [
            WebhookDelivery.model_validate(self._store.get(DELIVERIES, key).value)
            for key in reversed(keys)
        ]
