"""Persist live stream evidence as offline timeline fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.multimodal_types import SyncedEvidence, TimelineCue, VideoTimelineFixture

logger = get_logger("researcher.livekit_archive")


def evidence_to_fixture(
    *,
    stream_id: str,
    title: str,
    curriculum_topic: str = "kids-plasma",
    evidence: list[SyncedEvidence],
    duration_sec: float | None = None,
) -> VideoTimelineFixture:
    """Convert live evidence rows into a reusable timeline fixture."""
    cues: list[TimelineCue] = []
    max_ts = 0.0
    for row in sorted(evidence, key=lambda item: item.timestamp_sec):
        max_ts = max(max_ts, row.timestamp_sec)
        cues.append(
            TimelineCue(
                timestamp_sec=row.timestamp_sec,
                transcript_phrase=row.transcript_excerpt[:120] or title,
                visual=row.visual,
                audio=row.audio,
                concept_hints=row.concept_hints,
            )
        )
    return VideoTimelineFixture(
        source_id=f"livekit_{stream_id}",
        title=title,
        duration_sec=duration_sec or max(max_ts + 30.0, 60.0),
        transcript_ref=f"livekit_{stream_id}.txt",
        curriculum_topic=curriculum_topic,
        cues=tuple(cues),
    )


def save_live_archive(
    fixture: VideoTimelineFixture,
    output_dir: Path | str,
) -> Path:
    """Write one archived live stream fixture to disk."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{fixture.source_id}_archive.json"
    payload = json.loads(fixture.model_dump_json())
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("archived live stream fixture %s cues=%d", output_path.name, len(fixture.cues))
    return output_path


def archive_evidence_rows(
    *,
    stream_id: str,
    title: str,
    curriculum_topic: str,
    evidence: list[SyncedEvidence],
    output_dir: Path | str,
) -> Path | None:
    """Archive live evidence when a stream ends or a worker session closes."""
    live_rows = [row for row in evidence if row.live_stream_id == stream_id or row.is_live]
    if not live_rows:
        live_rows = list(evidence)
    if not live_rows:
        return None
    fixture = evidence_to_fixture(
        stream_id=stream_id,
        title=title,
        curriculum_topic=curriculum_topic,
        evidence=live_rows,
    )
    return save_live_archive(fixture, output_dir)
