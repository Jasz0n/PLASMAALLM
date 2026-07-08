"""Audio feature extraction for synced workshop evidence — stub offline, ffmpeg when available."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.researcher.multimodal_types import AudioCue, SyncedEvidence
from allm.researcher.video_decoder import ffmpeg_available

logger = get_logger("researcher.audio_analysis")

_MACHINE_TAGS = frozenset(
    {
        "machine-sound",
        "magnet-click",
        "click",
        "mechanical",
        "motor",
        "hum",
    }
)
_RHYTHM_TAGS = frozenset(
    {
        "field-beat",
        "beat",
        "rhythm",
        "rotation",
        "pulse",
    }
)
_PITCH_TAGS = frozenset(
    {
        "pitch",
        "tone",
        "whine",
        "resonance",
    }
)

_VOLUME_MEAN = re.compile(r"mean_volume:\s*([-\d.]+)\s*dB")
_VOLUME_MAX = re.compile(r"max_volume:\s*([-\d.]+)\s*dB")


class AudioFeatures(BaseModel):
    """Structured audio observations for one timeline moment."""

    model_config = ConfigDict(frozen=True)

    features: tuple[str, ...] = ()
    analysis: str = ""
    mean_volume_db: float | None = None
    max_volume_db: float | None = None


@runtime_checkable
class AudioAnalyzer(Protocol):
    """Derive audio features for one workshop moment."""

    def analyze(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        clip_path: str | None = None,
    ) -> AudioFeatures: ...


def _features_from_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    normalized = {tag.lower().replace("_", "-") for tag in tags}
    features: list[str] = []
    if normalized & _MACHINE_TAGS:
        features.append("machine_sound")
    if normalized & _RHYTHM_TAGS:
        features.append("rhythmic_beat")
    if normalized & _PITCH_TAGS:
        features.append("pitch_change")
    for tag in tags:
        token = tag.lower().replace(" ", "-")
        if token not in features and token not in {"machine-sound", "field-beat"}:
            features.append(token)
    return tuple(dict.fromkeys(features))


def _features_from_volume(mean_db: float | None, max_db: float | None) -> tuple[str, ...]:
    if mean_db is None:
        return ()
    features: list[str] = []
    if mean_db > -22.0:
        features.append("loud_impact")
    elif mean_db < -38.0:
        features.append("quiet_ambient")
    if max_db is not None and mean_db is not None and (max_db - mean_db) >= 12.0:
        features.append("dynamic_pulse")
    return tuple(features)


def _probe_volume(clip_path: Path) -> tuple[float | None, float | None]:
    if not ffmpeg_available() or not clip_path.is_file():
        return None, None
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(clip_path),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ffmpeg volumedetect error: %s", exc)
        return None, None
    stderr = result.stderr or ""
    mean_match = _VOLUME_MEAN.search(stderr)
    max_match = _VOLUME_MAX.search(stderr)
    mean_db = float(mean_match.group(1)) if mean_match else None
    max_db = float(max_match.group(1)) if max_match else None
    return mean_db, max_db


class StubAudioAnalyzer:
    """Offline analyzer using fixture tags and transcript context."""

    def analyze(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        clip_path: str | None = None,
    ) -> AudioFeatures:
        features = _features_from_tags(tags)
        excerpt = transcript_excerpt.strip()[:100]
        prefix = "Clip captured" if clip_path else "Transcript-aligned audio"
        parts = [prefix, description.strip()[:120]]
        if features:
            parts.append(f"features: {', '.join(features[:5])}")
        if excerpt:
            parts.append(f"context: {excerpt}")
        return AudioFeatures(features=features, analysis=" — ".join(part for part in parts if part))


class FfmpegAudioAnalyzer:
    """Merge tag heuristics with ffmpeg volumedetect when a clip exists."""

    def __init__(self, fallback: AudioAnalyzer | None = None) -> None:
        self._fallback = fallback or StubAudioAnalyzer()

    def analyze(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        clip_path: str | None = None,
    ) -> AudioFeatures:
        base = self._fallback.analyze(
            description=description,
            tags=tags,
            transcript_excerpt=transcript_excerpt,
            clip_path=clip_path,
        )
        path = Path(clip_path) if clip_path else None
        if path is None or not path.is_file():
            return base
        mean_db, max_db = _probe_volume(path)
        volume_features = _features_from_volume(mean_db, max_db)
        merged_features = tuple(dict.fromkeys(base.features + volume_features))
        volume_note = ""
        if mean_db is not None:
            volume_note = f"mean {mean_db:.1f} dB"
            if max_db is not None:
                volume_note += f", peak {max_db:.1f} dB"
        analysis = base.analysis
        if volume_note:
            analysis = f"{analysis} | {volume_note}"
        return base.model_copy(
            update={
                "features": merged_features,
                "analysis": analysis,
                "mean_volume_db": mean_db,
                "max_volume_db": max_db,
            }
        )


def get_audio_analyzer(backend: str = "auto") -> AudioAnalyzer:
    """Resolve audio analyzer: stub, ffmpeg, or auto (ffmpeg when ffmpeg exists)."""
    normalized = backend.lower().strip()
    if normalized in {"stub", "metadata", "offline"}:
        return StubAudioAnalyzer()
    if normalized in {"ffmpeg", "auto", "hybrid"}:
        if normalized == "auto" and not ffmpeg_available():
            return StubAudioAnalyzer()
        return FfmpegAudioAnalyzer(StubAudioAnalyzer())
    raise ValueError(f"unsupported audio analysis backend: {backend}")


def enrich_audio_cue(
    audio: AudioCue,
    *,
    analyzer: AudioAnalyzer,
    transcript_excerpt: str = "",
    clip_path: str | None = None,
) -> AudioCue:
    """Return audio cue with extracted features and analysis text."""
    path = clip_path or audio.clip_path
    features = analyzer.analyze(
        description=audio.description,
        tags=audio.tags,
        transcript_excerpt=transcript_excerpt,
        clip_path=path,
    )
    description = audio.description
    if features.analysis and features.analysis not in description:
        description = f"{description} | {features.analysis}"
    return audio.model_copy(
        update={
            "clip_path": path,
            "features": features.features,
            "analysis": features.analysis,
            "description": description,
        }
    )


def enrich_synced_evidence_audio(
    row: SyncedEvidence,
    *,
    analyzer: AudioAnalyzer,
    clip_path: str | None = None,
) -> SyncedEvidence:
    """Attach audio features and optional clip path to one synced evidence row."""
    if row.audio is None:
        return row
    audio = enrich_audio_cue(
        row.audio,
        analyzer=analyzer,
        transcript_excerpt=row.transcript_excerpt,
        clip_path=clip_path,
    )
    bonus = 0.0
    if audio.analysis:
        bonus += 0.04
    if audio.features:
        bonus += 0.03
    if audio.clip_path:
        bonus += 0.02
    confidence = min(1.0, round(row.confidence + bonus, 4))
    return row.model_copy(update={"audio": audio, "confidence": confidence})
