"""Versioned LoRA adapter storage.

Each fine-tune appends a new adapter record (never overwritten) so
exam provenance can reference which weights produced an answer.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from allm.storage.base import RecordStore

NAMESPACE = "adapters"
SEP = "::"


class AdapterRecord(BaseModel):
    """Metadata for one saved LoRA adapter."""

    model_config = ConfigDict(frozen=True)

    adapter_id: str
    student_id: str
    path: str
    base_model_id: str
    samples_trained: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdapterStore:
    """Filesystem adapters with versioned metadata in the record store."""

    def __init__(self, store: RecordStore, root: Path | str) -> None:
        self._store = store
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def scratch_dir(self, student_id: str) -> Path:
        """Temporary directory for in-flight training artifacts."""
        path = self.root / student_id / "_training"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        student_id: str,
        source_dir: Path,
        *,
        base_model_id: str,
        samples_trained: int,
        reason: str,
    ) -> str:
        """Copy adapter weights into the store and append metadata."""
        version = len(self.history(student_id)) + 1
        adapter_id = f"{student_id}-lora-{version:04d}"
        dest = self.root / student_id / adapter_id
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_dir, dest)
        record = AdapterRecord(
            adapter_id=adapter_id,
            student_id=student_id,
            path=str(dest),
            base_model_id=base_model_id,
            samples_trained=samples_trained,
        )
        self._store.put(
            NAMESPACE,
            f"{student_id}{SEP}{adapter_id}",
            json.loads(record.model_dump_json()),
            reason=reason,
        )
        return adapter_id

    def latest(self, student_id: str) -> AdapterRecord | None:
        """Most recent adapter for a student, if any."""
        history = self.history(student_id)
        return history[-1] if history else None

    def history(self, student_id: str) -> list[AdapterRecord]:
        """All adapters for a student, oldest first."""
        prefix = f"{student_id}{SEP}"
        records = []
        for key in self._store.keys(NAMESPACE):
            if key.startswith(prefix):
                raw = self._store.get(NAMESPACE, key)
                if raw is not None:
                    records.append(AdapterRecord.model_validate(raw.value))
        records.sort(key=lambda r: r.created_at)
        return records
