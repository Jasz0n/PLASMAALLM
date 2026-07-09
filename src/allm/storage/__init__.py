"""Versioned record storage.

Plan.md: "Long-term memory should never be overwritten blindly."
The storage layer enforces this at the lowest level: a put never
replaces data, it appends a new version. Memory, knowledge graph and
confidence history (later phases) all build on this guarantee.
"""

from allm.storage.base import NamespaceStat, Record, RecordStore, storage_backends
from allm.storage.sqlite import SQLiteRecordStore

__all__ = [
    "NamespaceStat",
    "Record",
    "RecordStore",
    "SQLiteRecordStore",
    "storage_backends",
]
