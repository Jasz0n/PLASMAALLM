"""Storage interface: append-only, versioned records.

Design decisions
----------------
- Records are addressed by ``(namespace, key)``; every write creates a
  new version instead of replacing the previous one, together with a
  ``reason`` — Plan.md requires a history of *why* beliefs changed.
- Values are plain JSON-serialisable dicts. Typed models (concepts,
  memories, exam results) are defined by the layers above; the store
  stays schema-agnostic so those layers can evolve independently.
- Implementations register in ``storage_backends`` and are chosen via
  configuration, so SQLite can later be swapped for PostgreSQL without
  touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from allm.core.registry import Registry


@dataclass(frozen=True)
class Record:
    """One immutable version of a stored value."""

    namespace: str
    key: str
    version: int
    value: dict[str, Any]
    reason: str | None
    created_at: datetime


@dataclass(frozen=True)
class NamespaceStat:
    """How much a namespace holds — the population census (M50 dashboard).

    A namespace that exists in the schema but reports zero keys is a
    subsystem that is wired but never exercised: exactly the "something
    is missing" signal an operator wants at a glance.
    """

    namespace: str
    keys: int  # distinct (namespace, key) pairs
    records: int  # total versions written
    last_write: datetime | None


@runtime_checkable
class RecordStore(Protocol):
    """Append-only versioned key/value store."""

    def put(
        self, namespace: str, key: str, value: dict[str, Any], *, reason: str | None = None
    ) -> Record:
        """Append a new version of ``(namespace, key)`` and return it."""
        ...

    def get(self, namespace: str, key: str) -> Record | None:
        """Return the latest version, or ``None`` if the key never existed."""
        ...

    def history(self, namespace: str, key: str) -> list[Record]:
        """Return all versions, oldest first."""
        ...

    def keys(self, namespace: str) -> list[str]:
        """Return all keys ever written in ``namespace``."""
        ...

    def namespaces(self) -> list[NamespaceStat]:
        """Per-namespace population census, busiest first.

        The append-only table read as a group-by: what the system has
        actually recorded, so a dashboard can show which subsystems are
        live and which are wired-but-empty without a hand-kept checklist.
        """
        ...

    def audit(
        self, namespace: str | None = None, *, limit: int = 100, offset: int = 0
    ) -> list[Record]:
        """The audit trail: every write, newest first (M50).

        The store is append-only, so this is not a separate log that
        could drift — it *is* the data, read in write order.
        """
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...


storage_backends: Registry[type] = Registry("storage_backend")
