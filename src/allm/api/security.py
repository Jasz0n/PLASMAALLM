"""API hardening: auth hook points and rate limiting (Roadmap M50).

The platform owns identity — the core only *verifies* what it is
handed. :class:`TokenVerifier` is the hook point: the platform injects
its own verifier (JWT, session lookup, ...) into ``create_app``; the
core ships two honest defaults:

- :class:`AllowAllVerifier` — development mode, loudly logged. Every
  request passes as the anonymous principal.
- :class:`StaticTokenVerifier` — one shared bearer token
  (``ALLM_API_TOKEN``), compared in constant time. Enough for a single
  trusted platform process; not a user-identity system, and it never
  pretends to be one.

Rate limiting is a per-principal token bucket, in-process and
dependency-free — the floor for a single-node deployment; a fronting
proxy owns the real thing in production.
"""

from __future__ import annotations

import hmac
import os
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from allm.core.logging import get_logger

logger = get_logger("api.security")


@dataclass(frozen=True)
class Principal:
    """Who the platform says is calling; opaque to the core."""

    contributor_id: str
    anonymous: bool = False


@runtime_checkable
class TokenVerifier(Protocol):
    """The auth hook point: token in, principal (or None) out."""

    def verify(self, token: str | None) -> Principal | None: ...


class AllowAllVerifier:
    """Development default — everything passes, and says so in the log."""

    def __init__(self) -> None:
        logger.warning(
            "API auth is OPEN (AllowAllVerifier) — set ALLM_API_TOKEN or "
            "inject a TokenVerifier before exposing this to strangers"
        )

    def verify(self, token: str | None) -> Principal | None:
        return Principal(contributor_id="anonymous", anonymous=True)


class StaticTokenVerifier:
    """One shared bearer token, constant-time compared."""

    def __init__(self, token: str) -> None:
        if not token or len(token) < 16:
            raise ValueError("ALLM_API_TOKEN must be at least 16 characters")
        self._token = token

    def verify(self, token: str | None) -> Principal | None:
        if token is None or not hmac.compare_digest(token, self._token):
            return None
        return Principal(contributor_id="platform")


@dataclass
class RateLimiter:
    """Token bucket per principal: ``requests`` per ``per_seconds``."""

    requests: int = 30
    per_seconds: float = 60.0
    _buckets: dict[str, tuple[float, float]] = field(default_factory=dict)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Consume one token for ``key``; False when the bucket is dry."""
        current = time.monotonic() if now is None else now
        tokens, last = self._buckets.get(key, (float(self.requests), current))
        tokens = min(
            float(self.requests),
            tokens + (current - last) * (self.requests / self.per_seconds),
        )
        if tokens < 1.0:
            self._buckets[key] = (tokens, current)
            return False
        self._buckets[key] = (tokens - 1.0, current)
        return True

    @classmethod
    def from_env(cls, raw: str | None) -> "RateLimiter":
        """Parse ``"30/60"`` (requests per seconds); defaults on None."""
        if not raw:
            return cls()
        requests, _, seconds = raw.partition("/")
        return cls(requests=int(requests), per_seconds=float(seconds or 60))


def default_verifier() -> TokenVerifier:
    """Environment-driven default: static token when configured, open otherwise."""
    token = os.environ.get("ALLM_API_TOKEN")
    if token:
        return StaticTokenVerifier(token)
    return AllowAllVerifier()
