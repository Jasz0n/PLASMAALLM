"""Observe LiveKit rooms — fixture snapshots offline, RTC when livekit is installed."""

from __future__ import annotations

import asyncio
import struct
import time
import wave
from pathlib import Path
from typing import Protocol, runtime_checkable

from allm.core.logging import get_logger
from allm.researcher.livekit_types import LiveKitCredentials, LiveMediaCapture, LiveStreamInfo
from allm.researcher.multimodal_types import AudioCue, SyncedEvidence, VisualCue

logger = get_logger("researcher.livekit_observer")


@runtime_checkable
class LiveKitObserver(Protocol):
    """Capture live workshop evidence from a LiveKit room."""

    def observe(
        self,
        stream: LiveStreamInfo,
        credentials: LiveKitCredentials,
        *,
        cache_dir: Path | str,
        capture_seconds: float = 3.0,
    ) -> list[SyncedEvidence]: ...


def _media_to_evidence(
    stream: LiveStreamInfo,
    captures: list[LiveMediaCapture],
) -> list[SyncedEvidence]:
    rows: list[SyncedEvidence] = []
    for capture in captures:
        visual = None
        audio = None
        if capture.frame_path:
            visual = VisualCue(
                description=f"Live frame from {capture.participant_identity or 'stream'}",
                frame_path=capture.frame_path,
                tags=("livekit", "live-stream"),
            )
        if capture.audio_clip_path:
            audio = AudioCue(
                description=f"Live audio from {capture.participant_identity or 'stream'}",
                clip_path=capture.audio_clip_path,
                tags=("livekit", "live-audio"),
            )
        rows.append(
            SyncedEvidence(
                source_id=f"livekit:{stream.stream_id}",
                timestamp_sec=capture.timestamp_sec,
                transcript_excerpt=stream.title,
                visual=visual,
                audio=audio,
                concept_hints=stream.tags,
                live_stream_id=stream.stream_id,
                is_live=True,
                confidence=0.85 if visual and audio else 0.82,
            )
        )
    return rows


def _snapshots_to_evidence(
    stream: LiveStreamInfo,
    *,
    cache_dir: Path,
    captures: list[LiveMediaCapture],
) -> list[SyncedEvidence]:
    del cache_dir
    if captures:
        return _media_to_evidence(stream, captures)

    rows: list[SyncedEvidence] = []
    for snapshot in stream.snapshots:
        rows.append(
            SyncedEvidence(
                source_id=f"livekit:{stream.stream_id}",
                timestamp_sec=snapshot.timestamp_sec,
                transcript_excerpt=snapshot.transcript_excerpt or stream.title,
                visual=snapshot.visual,
                audio=snapshot.audio,
                concept_hints=snapshot.concept_hints or stream.tags,
                live_stream_id=stream.stream_id,
                is_live=True,
                confidence=0.8 if snapshot.visual else 0.72,
            )
        )
    return rows


class StubLiveKitObserver:
    """Offline observer using fixture snapshots (no RTC connection)."""

    def observe(
        self,
        stream: LiveStreamInfo,
        credentials: LiveKitCredentials,
        *,
        cache_dir: Path | str,
        capture_seconds: float = 3.0,
    ) -> list[SyncedEvidence]:
        del credentials, capture_seconds
        return _snapshots_to_evidence(stream, cache_dir=Path(cache_dir), captures=[])


class LiveKitRtcObserver:
    """Connect to LiveKit via the Python RTC SDK when available."""

    def observe(
        self,
        stream: LiveStreamInfo,
        credentials: LiveKitCredentials,
        *,
        cache_dir: Path | str,
        capture_seconds: float = 3.0,
    ) -> list[SyncedEvidence]:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        try:
            captures = asyncio.run(
                _capture_rtc_media(credentials, cache, max_seconds=capture_seconds)
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("livekit RTC observe failed for %s: %s", stream.stream_id, exc)
            captures = []
        rows = _snapshots_to_evidence(stream, cache_dir=cache, captures=captures)
        if rows:
            return rows
        return StubLiveKitObserver().observe(
            stream,
            credentials,
            cache_dir=cache,
            capture_seconds=capture_seconds,
        )


async def _capture_rtc_media(
    credentials: LiveKitCredentials,
    cache_dir: Path,
    *,
    max_seconds: float,
) -> list[LiveMediaCapture]:
    """Subscribe to remote tracks and capture video frames and audio clips."""
    from livekit import rtc

    room = rtc.Room()
    captures: list[LiveMediaCapture] = []
    started = time.perf_counter()
    stop_at = started + max_seconds
    media_by_participant: dict[str, LiveMediaCapture] = {}

    async def _save_video(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        video_stream = rtc.VideoStream(track)
        async for event in video_stream:
            if time.perf_counter() >= stop_at:
                break
            frame = event.frame
            path = cache_dir / credentials.stream_id / f"live_{participant.identity}_{int(time.time())}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            saved = _write_video_frame(frame, path)
            if saved is None:
                continue
            ts = round(time.perf_counter() - started, 2)
            row = media_by_participant.get(participant.identity)
            if row is None:
                row = LiveMediaCapture(
                    stream_id=credentials.stream_id,
                    timestamp_sec=ts,
                    participant_identity=participant.identity,
                )
                media_by_participant[participant.identity] = row
            media_by_participant[participant.identity] = row.model_copy(
                update={"frame_path": str(saved), "timestamp_sec": ts}
            )
            if len(media_by_participant) >= max(1, int(max_seconds)):
                break

    async def _save_audio(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        audio_stream = rtc.AudioStream(track)
        pcm_chunks: list[bytes] = []
        sample_rate = 48000
        num_channels = 1
        async for event in audio_stream:
            if time.perf_counter() >= stop_at:
                break
            frame = event.frame
            sample_rate = frame.sample_rate
            num_channels = frame.num_channels
            pcm_chunks.append(frame.data)
        if not pcm_chunks:
            return
        path = cache_dir / credentials.stream_id / f"live_audio_{participant.identity}_{int(time.time())}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        saved = _write_wav(path, b"".join(pcm_chunks), sample_rate, num_channels)
        if saved is None:
            return
        ts = round(time.perf_counter() - started, 2)
        row = media_by_participant.get(participant.identity)
        if row is None:
            row = LiveMediaCapture(
                stream_id=credentials.stream_id,
                timestamp_sec=ts,
                participant_identity=participant.identity,
            )
        media_by_participant[participant.identity] = row.model_copy(
            update={"audio_clip_path": str(saved), "timestamp_sec": ts}
        )

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        del publication
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(_save_video(track, participant))
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(_save_audio(track, participant))

    await room.connect(credentials.url, credentials.token)
    try:
        while time.perf_counter() < stop_at and len(media_by_participant) < max(1, int(max_seconds)):
            await asyncio.sleep(0.2)
    finally:
        await room.disconnect()

    captures.extend(media_by_participant.values())
    return captures


def _write_video_frame(frame, output_path: Path) -> Path | None:
    """Persist one RGBA video frame as JPEG when Pillow is available."""
    try:
        from livekit import rtc
        from PIL import Image
    except ImportError:
        return None
    try:
        rgba = frame.convert(rtc.VideoBufferType.RGBA)
        image = Image.frombytes("RGBA", (rgba.width, rgba.height), rgba.data)
        image.convert("RGB").save(output_path, format="JPEG", quality=85)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("failed to write livekit frame: %s", exc)
        return None
    return output_path if output_path.is_file() else None


def _write_wav(
    output_path: Path,
    pcm_data: bytes,
    sample_rate: int,
    num_channels: int,
) -> Path | None:
    """Write raw PCM bytes to a WAV file."""
    if not pcm_data:
        return None
    try:
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(max(1, num_channels))
            handle.setsampwidth(2)
            handle.setframerate(max(8000, sample_rate))
            handle.writeframes(pcm_data)
    except (OSError, struct.error, wave.Error) as exc:
        logger.warning("failed to write livekit audio clip: %s", exc)
        return None
    return output_path if output_path.is_file() else None


def get_livekit_observer(backend: str = "auto") -> LiveKitObserver:
    """Resolve observer: stub, rtc, or auto (RTC when livekit import succeeds)."""
    normalized = backend.lower().strip()
    if normalized in {"stub", "offline", "fixture"}:
        return StubLiveKitObserver()
    if normalized in {"rtc", "livekit"}:
        return LiveKitRtcObserver()
    if normalized in {"auto", "hybrid"}:
        try:
            import livekit.rtc  # noqa: F401

            return LiveKitRtcObserver()
        except ImportError:
            logger.info("livekit RTC SDK not installed — using stub livekit observer")
            return StubLiveKitObserver()
    raise ValueError(f"unsupported livekit observer backend: {backend}")
