"""Persistent LiveKit observer buffer for ongoing workshop streams."""

from __future__ import annotations

import threading
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.livekit_archive import archive_evidence_rows
from allm.researcher.multimodal_types import SyncedEvidence

logger = get_logger("researcher.livekit_worker")

_worker_lock = threading.Lock()
_worker_instance: "LiveKitObserverWorker | None" = None


class LiveKitObserverWorker:
    """Accumulate live evidence across Researcher cycles for one or more streams."""

    def __init__(self, cache_dir: Path | str) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: dict[str, list[SyncedEvidence]] = {}
        self._titles: dict[str, str] = {}
        self._topics: dict[str, str] = {}

    def append(
        self,
        stream_id: str,
        rows: list[SyncedEvidence],
        *,
        title: str = "",
        curriculum_topic: str = "kids-plasma",
    ) -> None:
        """Buffer evidence captured from one observation pass."""
        if not rows:
            return
        bucket = self._buffer.setdefault(stream_id, [])
        bucket.extend(rows)
        if title:
            self._titles[stream_id] = title
        if curriculum_topic:
            self._topics[stream_id] = curriculum_topic
        logger.info(
            "livekit worker buffered stream=%s total=%d added=%d",
            stream_id,
            len(bucket),
            len(rows),
        )

    def buffered(self, stream_id: str) -> list[SyncedEvidence]:
        """Return buffered evidence for one stream."""
        return list(self._buffer.get(stream_id, []))

    def stream_ids(self) -> tuple[str, ...]:
        """Return stream ids with buffered evidence."""
        return tuple(self._buffer.keys())

    def archive_stream(
        self,
        stream_id: str,
        *,
        archive_dir: Path | str | None = None,
    ) -> Path | None:
        """Write buffered evidence to an offline fixture and clear the buffer."""
        rows = self._buffer.get(stream_id, [])
        if not rows:
            return None
        target = Path(archive_dir) if archive_dir is not None else self._cache_dir / "archives"
        path = archive_evidence_rows(
            stream_id=stream_id,
            title=self._titles.get(stream_id, stream_id),
            curriculum_topic=self._topics.get(stream_id, "kids-plasma"),
            evidence=rows,
            output_dir=target,
        )
        self._buffer.pop(stream_id, None)
        return path

    def archive_all(self, *, archive_dir: Path | str | None = None) -> list[Path]:
        """Archive every buffered stream."""
        paths: list[Path] = []
        for stream_id in list(self._buffer.keys()):
            archived = self.archive_stream(stream_id, archive_dir=archive_dir)
            if archived is not None:
                paths.append(archived)
        return paths


def get_livekit_worker(cache_dir: Path | str) -> LiveKitObserverWorker:
    """Return a process-wide worker singleton."""
    global _worker_instance
    with _worker_lock:
        if _worker_instance is None:
            _worker_instance = LiveKitObserverWorker(cache_dir)
        return _worker_instance


def reset_livekit_worker() -> None:
    """Clear the worker singleton (tests)."""
    global _worker_instance
    with _worker_lock:
        _worker_instance = None
