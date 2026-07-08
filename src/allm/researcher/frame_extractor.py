"""Extract still frames from workshop videos via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.video_decoder import ffmpeg_available, match_video_for_transcript

logger = get_logger("researcher.frame_extractor")


def frame_output_path(
    cache_dir: Path | str,
    source_id: str,
    timestamp_sec: float,
) -> Path:
    """Deterministic path for one extracted frame."""
    safe_ts = str(timestamp_sec).replace(".", "_")
    return Path(cache_dir) / source_id / f"frame_{safe_ts}.jpg"


def extract_frame_at_timestamp(
    video_path: Path | str,
    timestamp_sec: float,
    output_path: Path | str,
) -> Path | None:
    """Extract one JPEG frame at ``timestamp_sec`` using ffmpeg."""
    if not ffmpeg_available():
        return None
    video = Path(video_path)
    output = Path(output_path)
    if not video.is_file():
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(max(0.0, timestamp_sec)),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg frame extract failed: %s", result.stderr[:200])
            return None
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg frame extract error: %s", exc)
        return None
    return output if output.is_file() else None


def extract_frame_for_source(
    *,
    source_id: str,
    transcript_name: str,
    timestamp_sec: float,
    video_dir: Path | str | None,
    cache_dir: Path | str,
) -> Path | None:
    """Resolve workshop video and extract one cached frame."""
    if video_dir is None:
        return None
    video_path = match_video_for_transcript(Path(transcript_name), Path(video_dir))
    if video_path is None:
        video_path = match_video_for_transcript(Path(f"{source_id}.txt"), Path(video_dir))
    if video_path is None:
        return None
    output = frame_output_path(cache_dir, source_id, timestamp_sec)
    if output.is_file():
        return output
    return extract_frame_at_timestamp(video_path, timestamp_sec, output)


def sample_timestamps_for_span(
    *,
    timestamp_sec: float,
    frame_start: int | None,
    frame_end: int | None,
    sample_count: int = 3,
    fps: float = 30.0,
) -> tuple[float, ...]:
    """Return evenly spaced timestamps across a visual cue span."""
    count = max(2, sample_count)
    if frame_start is not None and frame_end is not None and frame_end > frame_start:
        span_sec = max((frame_end - frame_start) / fps, 0.25)
        start = max(0.0, timestamp_sec)
        if count == 1:
            return (start,)
        step = span_sec / (count - 1)
        return tuple(round(start + (index * step), 3) for index in range(count))
    return tuple(round(timestamp_sec + (index * 0.4), 3) for index in range(count))


def extract_frame_sequence_for_source(
    *,
    source_id: str,
    transcript_name: str,
    timestamp_sec: float,
    frame_start: int | None,
    frame_end: int | None,
    video_dir: Path | str | None,
    cache_dir: Path | str,
    sample_count: int = 3,
    fps: float = 30.0,
) -> tuple[Path, ...]:
    """Extract a short frame sequence for temporal motion analysis."""
    timestamps = sample_timestamps_for_span(
        timestamp_sec=timestamp_sec,
        frame_start=frame_start,
        frame_end=frame_end,
        sample_count=sample_count,
        fps=fps,
    )
    paths: list[Path] = []
    for ts in timestamps:
        extracted = extract_frame_for_source(
            source_id=source_id,
            transcript_name=transcript_name,
            timestamp_sec=ts,
            video_dir=video_dir,
            cache_dir=cache_dir,
        )
        if extracted is not None:
            paths.append(extracted)
    return tuple(paths)
