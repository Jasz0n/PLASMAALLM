"""File-based experiment tracker.

Each run gets its own directory under the tracking root:

    <root>/<run_id>/
        meta.json      name, status, timestamps
        params.json    logged parameters (merged across calls)
        metrics.jsonl  one JSON object per metric observation
        artifacts/     named text artifacts

Plain files keep runs greppable and diffable, which matters more in a
research setting than dashboards. A hosted tracker can be added later
as another ``tracker_backends`` entry.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from allm.tracking.base import tracker_backends


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalRun:
    """A run backed by a directory. Created via :class:`LocalTracker`."""

    def __init__(self, root: Path, run_id: str, name: str) -> None:
        self._dir = root / run_id
        self._dir.mkdir(parents=True)
        (self._dir / "artifacts").mkdir()
        self._run_id = run_id
        self._finished = False
        self._params: dict[str, Any] = {}
        self._write_meta({"name": name, "status": "running", "started_at": _utcnow()})

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def directory(self) -> Path:
        return self._dir

    def log_params(self, params: dict[str, Any]) -> None:
        self._check_open()
        self._params.update(params)
        (self._dir / "params.json").write_text(
            json.dumps(self._params, indent=2, default=str), encoding="utf-8"
        )

    def log_metric(self, name: str, value: float, *, step: int | None = None) -> None:
        self._check_open()
        entry = {"time": _utcnow(), "name": name, "value": float(value), "step": step}
        with (self._dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def log_artifact(self, name: str, content: str) -> None:
        self._check_open()
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        (self._dir / "artifacts" / safe).write_text(content, encoding="utf-8")

    def finish(self, status: str = "completed") -> None:
        self._check_open()
        meta = json.loads((self._dir / "meta.json").read_text(encoding="utf-8"))
        meta.update({"status": status, "finished_at": _utcnow()})
        self._write_meta(meta)
        self._finished = True

    def _write_meta(self, meta: dict[str, Any]) -> None:
        (self._dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError(f"run {self._run_id} is already finished")


@tracker_backends.register("local")
class LocalTracker:
    """Creates :class:`LocalRun` directories under ``root``."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def start_run(self, name: str) -> LocalRun:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        slug = re.sub(r"[^A-Za-z0-9_-]", "-", name)[:40] or "run"
        return LocalRun(self._root, f"{stamp}-{slug}", name)

    def list_runs(self) -> list[str]:
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())
