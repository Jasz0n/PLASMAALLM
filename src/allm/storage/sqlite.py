"""SQLite implementation of the versioned record store.

SQLite is the default backend because it is zero-setup and perfectly
adequate for single-machine experiments. The schema is intentionally
trivial — one append-only table — so migrating to PostgreSQL later is
a copy job, not a redesign.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from allm.storage.base import NamespaceStat, Record, storage_backends

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    version    INTEGER NOT NULL,
    value      TEXT NOT NULL,
    reason     TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key, version)
);
"""


@storage_backends.register("sqlite")
class SQLiteRecordStore:
    """Append-only versioned store backed by a single SQLite file.

    Thread-safe: SQLite connections are per-thread guarded by a lock,
    which is enough for the coarse write patterns of the learning loop.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def put(
        self, namespace: str, key: str, value: dict[str, Any], *, reason: str | None = None
    ) -> Record:
        created_at = datetime.now(timezone.utc)
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM records WHERE namespace=? AND key=?",
                (namespace, key),
            ).fetchone()
            version = int(row[0]) + 1
            self._conn.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
                (namespace, key, version, json.dumps(value), reason, created_at.isoformat()),
            )
            self._conn.commit()
        return Record(namespace, key, version, value, reason, created_at)

    def get(self, namespace: str, key: str) -> Record | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT version, value, reason, created_at FROM records"
                " WHERE namespace=? AND key=? ORDER BY version DESC LIMIT 1",
                (namespace, key),
            ).fetchone()
        return self._row_to_record(namespace, key, row) if row else None

    def history(self, namespace: str, key: str) -> list[Record]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT version, value, reason, created_at FROM records"
                " WHERE namespace=? AND key=? ORDER BY version ASC",
                (namespace, key),
            ).fetchall()
        return [self._row_to_record(namespace, key, row) for row in rows]

    def keys(self, namespace: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT key FROM records WHERE namespace=? ORDER BY key",
                (namespace,),
            ).fetchall()
        return [row[0] for row in rows]

    def namespaces(self) -> list[NamespaceStat]:
        """Per-namespace census, busiest first — one grouped read."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT namespace, COUNT(DISTINCT key), COUNT(*), MAX(created_at)"
                " FROM records GROUP BY namespace ORDER BY COUNT(*) DESC, namespace ASC"
            ).fetchall()
        return [
            NamespaceStat(
                namespace=row[0],
                keys=int(row[1]),
                records=int(row[2]),
                last_write=datetime.fromisoformat(row[3]) if row[3] else None,
            )
            for row in rows
        ]

    def audit(
        self, namespace: str | None = None, *, limit: int = 100, offset: int = 0
    ) -> list[Record]:
        """Every write, newest first — the append-only table read backwards."""
        query = (
            "SELECT namespace, key, version, value, reason, created_at FROM records"
        )
        params: list[Any] = []
        if namespace is not None:
            query += " WHERE namespace=?"
            params.append(namespace)
        query += " ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            self._row_to_record(row[0], row[1], row[2:]) for row in rows
        ]

    def backup_to(self, destination: Path | str) -> Path:
        """Consistent online backup via SQLite's backup API."""
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(dest)
            try:
                self._conn.backup(target)
                target.commit()
            finally:
                target.close()
        return dest

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_record(namespace: str, key: str, row: tuple) -> Record:
        version, value, reason, created_at = row
        return Record(
            namespace=namespace,
            key=key,
            version=int(version),
            value=json.loads(value),
            reason=reason,
            created_at=datetime.fromisoformat(created_at),
        )
