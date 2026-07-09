"""A small, typed HTTP client for the ALLM API (Roadmap M52).

So an integrator — starting with the frontend team — can drive the whole
contributor loop without reading engine source or hand-rolling requests.
Zero dependencies (stdlib ``urllib``), aligned to the frozen wire
contract, and it turns the JSON error envelope into a typed exception.

The transport is injectable (same pattern as the webhook sender): the
default speaks real HTTP, and a test or example can pass one backed by an
in-process app. This module is also the reference a non-Python client
(the browser app) reads to see exactly what calls the loop needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


class AllmError(RuntimeError):
    """A non-2xx response, carrying the API's structured error envelope."""

    def __init__(self, status: int, message: str, type: str = "http_error", fields: Any = None):
        super().__init__(f"{status} {type}: {message}")
        self.status = status
        self.message = message
        self.type = type
        self.fields = fields

    @classmethod
    def from_response(cls, response: "Response") -> "AllmError":
        try:
            error = response.json().get("error", {})
        except Exception:
            error = {}
        return cls(
            status=response.status,
            message=error.get("message", response.text[:200] or "request failed"),
            type=error.get("type", "http_error"),
            fields=error.get("fields"),
        )


@dataclass
class Response:
    """The minimal shape a transport returns."""

    status: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.body) if self.body else None

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")


class Transport(Protocol):
    """Send one request. Inject a fake to test without a network."""

    def request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> Response: ...


class UrllibTransport:
    """Default transport — stdlib ``urllib``, no dependencies."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def request(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> Response:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return Response(response.status, response.read(), dict(response.headers))
        except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a JSON envelope
            return Response(exc.code, exc.read(), dict(exc.headers or {}))


class AllmClient:
    """Typed access to one ALLM instance.

    Reads need no token; writes require the bearer token issued by the
    operator. Every non-2xx raises :class:`AllmError` with the envelope's
    ``status`` / ``type`` / ``message`` / ``fields``.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        token: str | None = None,
        *,
        transport: Transport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._transport = transport or UrllibTransport()

    # -- plumbing -------------------------------------------------------

    def _call(
        self, method: str, path: str, *, body: Any = None, auth: bool = False
    ) -> Any:
        headers = {"Accept": "application/json"}
        payload: bytes | None = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if auth:
            if not self._token:
                raise AllmError(401, "this call needs a token; construct AllmClient(token=…)", "auth")
            headers["Authorization"] = f"Bearer {self._token}"
        response = self._transport.request(method, self._base + path, headers, payload)
        if response.status >= 400:
            raise AllmError.from_response(response)
        return response.json()

    # -- health & contract ----------------------------------------------

    def health(self) -> dict:
        return self._call("GET", "/health")

    def ready(self) -> dict:
        return self._call("GET", "/ready")

    def wire(self) -> dict:
        """The frozen wire contract — build against this, not our source."""
        return self._call("GET", "/wire")

    # -- knowledge (open reads) -----------------------------------------

    def concepts(self, *, limit: int = 100, offset: int = 0) -> list[dict]:
        return self._call("GET", f"/concepts?limit={limit}&offset={offset}")

    def concept(self, name: str) -> dict:
        """One concept with its provenance and evidential confidence."""
        from urllib.parse import quote

        return self._call("GET", f"/concepts/{quote(name)}")

    def ask(self, query: str) -> dict:
        """Grounded Q&A: an answer from the evidence graph, with confidence
        and provenance — or an honest 'no evidence yet'. Never a guess."""
        from urllib.parse import quote

        return self._call("GET", f"/ask?q={quote(query)}")

    # -- contributions (writes) -----------------------------------------

    def submit_evidence(
        self,
        *,
        claim: str,
        concept: str,
        contributor: str,
        outcome: str,
        kind: str = "experiment",
        **extra: Any,
    ) -> dict:
        """Submit one evidence package; returns its id + confidence breakdown."""
        body = {
            "claim": claim, "concept": concept, "contributor": contributor,
            "outcome": outcome, "kind": kind, **extra,
        }
        return self._call("POST", "/evidence", body=body, auth=True)

    def submit_documents(self, documents: list[dict]) -> dict:
        """Ingest raw explanation streams; conflicts auto-open proposals."""
        return self._call("POST", "/documents", body=documents, auth=True)

    # -- proposals ------------------------------------------------------

    def proposals(self, *, status: str | None = None) -> list[dict]:
        path = "/proposals" + (f"?status={status}" if status else "")
        return self._call("GET", path)

    def claim_proposal(self, proposal_id: str, contributor: str) -> dict:
        return self._call(
            "POST", f"/proposals/{proposal_id}/claim",
            body={"contributor": contributor}, auth=True,
        )

    def resolve_proposal(self, proposal_id: str, packages: list[dict]) -> dict:
        return self._call(
            "POST", f"/proposals/{proposal_id}/resolve",
            body={"packages": packages}, auth=True,
        )

    # -- the live feed --------------------------------------------------

    def events(self, *, since: int = 0, limit: int = 100) -> dict:
        """One page of the feed: ``{events, cursor, total}``."""
        return self._call("GET", f"/events?since={since}&limit={limit}")

    def catch_up(self, *, since: int = 0) -> list[dict]:
        """Drain every event after ``since`` (paginates to the end)."""
        collected: list[dict] = []
        cursor = since
        while True:
            page = self.events(since=cursor, limit=200)
            batch = page["events"]
            if not batch:
                return collected
            collected.extend(batch)
            cursor = page["cursor"]

    def audit(self, *, limit: int = 100) -> list[dict]:
        return self._call("GET", f"/audit?limit={limit}")

    # -- webhooks (platform-side, approval-gated) -----------------------

    def register_webhook(
        self, url: str, *, event_types: list[str] | None = None, secret: str | None = None
    ) -> dict:
        """Register a delivery endpoint (proposed; the secret returns once)."""
        body: dict[str, Any] = {"url": url}
        if event_types is not None:
            body["event_types"] = event_types
        if secret is not None:
            body["secret"] = secret
        return self._call("POST", "/webhooks", body=body, auth=True)

    def approve_webhook(self, subscription_id: str, *, approver: str, reason: str) -> dict:
        return self._call(
            "POST", f"/webhooks/{subscription_id}/approve",
            body={"approver": approver, "reason": reason}, auth=True,
        )
