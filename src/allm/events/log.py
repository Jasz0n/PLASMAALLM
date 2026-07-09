"""An ordered, append-only event stream over the record store (M51).

Events are the platform's live feed. Each carries a monotonic ``seq`` so
a subscriber can poll ``since(cursor)`` and never miss or replay one —
the frontend's "what changed since I last looked". Kept deliberately
thin: the core emits facts (a proposal opened, a confidence moved);
*acting* on them — notifications, webhooks, rewards — is the platform's
job, exactly like identity and incentives.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.storage.base import RecordStore

logger = get_logger("events")

NAMESPACE = "events"


class Event(BaseModel):
    """One thing that happened, in order."""

    model_config = ConfigDict(frozen=True)

    seq: int  # monotonic, 1-based; the subscriber's cursor
    type: str  # dotted, e.g. "confidence.changed", "proposal.opened"
    subject: str  # the primary id/name this event is about
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class EventLog:
    """Append-only, ordered stream of domain events over one store."""

    def __init__(
        self, store: RecordStore, *, on_emit: Callable[[Event], Any] | None = None
    ) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._on_emit = on_emit

    def emit(self, type: str, subject: str, data: dict[str, Any] | None = None) -> Event:
        """Append one event and return it with its assigned ``seq``.

        An ``on_emit`` hook (e.g. webhook dispatch) fires *after* the
        write and *outside* the lock, so slow delivery never serialises
        emits, and a failing subscriber can never break the core write.
        """
        with self._lock:
            seq = len(self._store.keys(NAMESPACE)) + 1
            event = Event(
                seq=seq,
                type=type,
                subject=subject,
                data=data or {},
                created_at=datetime.now(timezone.utc),
            )
            self._store.put(
                NAMESPACE,
                f"{seq:012d}",
                json.loads(event.model_dump_json()),
                reason=f"{type}: {subject}",
            )
        logger.info("event #%d %s %s", seq, type, subject)
        if self._on_emit is not None:
            try:
                self._on_emit(event)
            except Exception:  # a subscriber must never break a core write
                logger.exception("on_emit hook failed for event #%d", seq)
        return event

    def since(self, cursor: int = 0, *, limit: int = 100) -> list[Event]:
        """Events with ``seq > cursor``, oldest first (the poll operation).

        ``cursor`` is the last ``seq`` the caller has already seen, so a
        subscriber advances it to the ``seq`` of the last returned event.
        Keys are zero-padded and stored in order, so ``keys[cursor:]`` is
        exactly the un-seen tail without scanning what came before.
        """
        keys = self._store.keys(NAMESPACE)[cursor : cursor + limit]
        return [self._read(key) for key in keys]

    def latest(self, *, limit: int = 50) -> list[Event]:
        """The most recent events, newest first — for a feed preview."""
        keys = self._store.keys(NAMESPACE)[-limit:]
        return [self._read(key) for key in reversed(keys)]

    def count(self) -> int:
        return len(self._store.keys(NAMESPACE))

    def _read(self, key: str) -> Event:
        record = self._store.get(NAMESPACE, key)
        return Event.model_validate(record.value)
