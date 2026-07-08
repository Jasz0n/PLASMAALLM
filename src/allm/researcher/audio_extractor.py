"""Extract short audio clips from workshop videos via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.video_decoder import ffmpeg_available, match_video_for_transcript

logger = get_logger("researcher.audio_extractor")

DEFAULT_CLIP_DURATION_SEC = 2.5


def clip_output_path(
    cache_dir: Path | str,
    source_id: str,
    timestamp_sec: float,
) -> Path:
    """Deterministic path for one extracted audio clip."""
    safe_ts = str(timestamp_sec).replace(".", "_")
    return Path(cache_dir) / source_id / f"audio_{safe_ts}.wav"


def extract_audio_clip_at_timestamp(
    video_path: Path | str,
    timestamp_sec: float,
    output_path: Path | str,
    *,
    duration_sec: float = DEFAULT_CLIP_DURATION_SEC,
) -> Path | None:
    """Extract a mono WAV clip at ``timestamp_sec`` using ffmpeg."""
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
                "-t",
                str(max(0.5, duration_sec)),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg audio extract failed: %s", result.stderr[:200])
            return None
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg audio extract error: %s", exc)
        return None
    return output if output.is_file() else None


def extract_audio_for_source(
    *,
    source_id: str,
    transcript_name: str,
    timestamp_sec: float,
    video_dir: Path | str | None,
    cache_dir: Path | str,
    duration_sec: float = DEFAULT_CLIP_DURATION_SEC,
) -> Path | None:
    """Resolve workshop video and extract one cached audio clip."""
    if video_dir is None:
        return None
    video_path = match_video_for_transcript(Path(transcript_name), Path(video_dir))
    if video_path is None:
        video_path = match_video_for_transcript(Path(f"{source_id}.txt"), Path(video_dir))
    if video_path is None:
        return None
    output = clip_output_path(cache_dir, source_id, timestamp_sec)
    if output.is_file():
        return output
    return extract_audio_clip_at_timestamp(
        video_path,
        timestamp_sec,
        output,
        duration_sec=duration_sec,
    )
