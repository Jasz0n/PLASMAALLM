"""Video decoder — probe MP4s and auto-generate timeline fixtures from transcripts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.multimodal import (
    VIDEO_MENTION_PHRASES,
    _excerpt_around_phrase,
    _normalize,
    load_video_fixture,
)
from allm.researcher.multimodal_types import TimelineCue, VideoTimelineFixture, VisualCue

logger = get_logger("researcher.video_decoder")

DEFAULT_FPS = 30.0
DEFAULT_DURATION_SEC = 4200.0
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".avi", ".mov")


def ffmpeg_available() -> bool:
    """Return True when ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    """Return True when ffprobe is on PATH."""
    return shutil.which("ffprobe") is not None


def probe_video_duration(video_path: Path | str) -> float | None:
    """Read video duration in seconds via ffprobe."""
    if not ffprobe_available():
        return None
    path = Path(video_path)
    if not path.is_file():
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return None


def match_video_for_transcript(transcript_path: Path, video_dir: Path | None) -> Path | None:
    """Find a video file matching a workshop transcript stem."""
    if video_dir is None or not video_dir.is_dir():
        return None
    stem = transcript_path.stem
    for ext in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    for path in sorted(video_dir.glob(f"{stem}.*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def find_video_mentions(text: str) -> list[tuple[str, int]]:
    """Return (phrase, char_offset) for each video reference in transcript text."""
    normalized = _normalize(text)
    mentions: list[tuple[str, int]] = []
    for phrase in VIDEO_MENTION_PHRASES:
        needle = _normalize(phrase)
        start = 0
        while True:
            index = normalized.find(needle, start)
            if index < 0:
                break
            mentions.append((phrase, index))
            start = index + max(1, len(needle))
    return sorted(mentions, key=lambda row: row[1])


def generate_fixture_from_transcript(
    transcript_path: Path | str,
    *,
    video_path: Path | str | None = None,
    curriculum_topic: str = "kids-plasma",
    default_duration_sec: float = DEFAULT_DURATION_SEC,
    fps: float = DEFAULT_FPS,
) -> VideoTimelineFixture | None:
    """Build a timeline fixture from transcript video mentions."""
    path = Path(transcript_path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    mentions = find_video_mentions(text)
    if not mentions:
        return None

    duration = default_duration_sec
    video = Path(video_path) if video_path is not None else None
    if video is not None and video.is_file():
        probed = probe_video_duration(video)
        if probed is not None:
            duration = probed

    normalized_len = max(1, len(_normalize(text)))
    cues: list[TimelineCue] = []
    for phrase, offset in mentions:
        timestamp = round((offset / normalized_len) * duration, 2)
        frame = int(timestamp * fps)
        excerpt = _excerpt_around_phrase(text, phrase, window=80)
        cues.append(
            TimelineCue(
                timestamp_sec=timestamp,
                transcript_phrase=phrase,
                visual=VisualCue(
                    description=f"Auto-detected video moment: {excerpt[:100]}",
                    frame_start=frame,
                    frame_end=frame + int(fps * 2),
                    tags=("auto-generated", "video-mention"),
                ),
                concept_hints=("plasma", "video demonstration"),
            )
        )

    source_id = path.stem
    return VideoTimelineFixture(
        source_id=source_id,
        title=f"Auto timeline for {source_id}",
        duration_sec=duration,
        transcript_ref=path.name,
        curriculum_topic=curriculum_topic,
        cues=tuple(cues),
    )


def save_video_fixture(fixture: VideoTimelineFixture, output_path: Path | str) -> Path:
    """Persist one fixture as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json.loads(fixture.model_dump_json()), indent=2),
        encoding="utf-8",
    )
    return path


def ensure_workshop_fixtures(
    workshop_dir: Path | str,
    fixture_dir: Path | str,
    *,
    video_dir: Path | str | None = None,
    curriculum_topic: str = "kids-plasma",
    overwrite: bool = False,
) -> list[VideoTimelineFixture]:
    """Generate missing timeline fixtures for transcripts with video mentions."""
    workshop = Path(workshop_dir)
    output = Path(fixture_dir)
    output.mkdir(parents=True, exist_ok=True)
    videos = Path(video_dir) if video_dir is not None else None

    fixtures: list[VideoTimelineFixture] = []
    existing_sources: set[str] = set()
    for path in output.glob("*.json"):
        if path.name.startswith("livekit_"):
            continue
        try:
            existing_sources.add(load_video_fixture(path).source_id)
        except (json.JSONDecodeError, ValueError):
            continue

    for transcript_path in sorted(workshop.glob("*.txt")):
        if not overwrite and transcript_path.stem in existing_sources:
            continue
        output_file = output / f"{transcript_path.stem}_auto.json"
        video_path = match_video_for_transcript(transcript_path, videos)
        fixture = generate_fixture_from_transcript(
            transcript_path,
            video_path=video_path,
            curriculum_topic=curriculum_topic,
        )
        if fixture is None:
            continue
        save_video_fixture(fixture, output_file)
        fixtures.append(fixture)
        logger.info(
            "generated fixture %s cues=%d video=%s",
            output_file.name,
            len(fixture.cues),
            video_path.name if video_path else "transcript-only",
        )
    return fixtures
