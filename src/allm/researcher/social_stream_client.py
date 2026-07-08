"""SocialServer client for discovering live LiveKit streams."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.livekit_types import LiveStreamInfo, LiveStreamSnapshot
from allm.researcher.multimodal_types import AudioCue, VisualCue

logger = get_logger("researcher.social_stream_client")


def _parse_stream_row(row: dict) -> LiveStreamInfo | None:
    stream_id = str(row.get("id") or row.get("stream_id") or "").strip()
    if not stream_id:
        return None
    room_name = (
        row.get("livekitRoomName")
        or row.get("livekit_room_name")
        or (row.get("recordingRoomInfo") or {}).get("roomName")
        or stream_id
    )
    livekit_url = (
        row.get("livekitUrl")
        or row.get("livekit_url")
        or (row.get("recordingRoomInfo") or {}).get("url")
        or os.environ.get("LIVEKIT_URL", "")
    )
    if not livekit_url:
        return None
    tags = tuple(str(tag) for tag in (row.get("tags") or []) if tag)
    return LiveStreamInfo(
        stream_id=stream_id,
        title=str(row.get("title") or stream_id),
        status=str(row.get("status") or "live"),
        livekit_room_name=str(room_name),
        livekit_url=str(livekit_url),
        curriculum_topic=str(row.get("topic") or row.get("curriculum_topic") or "kids-plasma"),
        topic=str(row.get("topic") or ""),
        tags=tags,
    )


def fetch_active_streams(base_url: str, *, timeout_sec: int = 10) -> list[LiveStreamInfo]:
    """List live streams from SocialServer ``GET /api/streams/active``."""
    root = base_url.rstrip("/")
    request = urllib.request.Request(f"{root}/api/streams/active", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        logger.warning("social active streams fetch failed: %s", exc)
        return []

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []

    streams: list[LiveStreamInfo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed = _parse_stream_row(row)
        if parsed is not None and parsed.status == "live":
            streams.append(parsed)
    return streams


def load_livekit_fixture(path: Path | str) -> list[LiveStreamInfo]:
    """Load offline live stream fixtures for development without SocialServer."""
    fixture_path = Path(path)
    if not fixture_path.is_file():
        return []
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows = payload.get("streams", payload if isinstance(payload, list) else [])
    streams: list[LiveStreamInfo] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        snapshots = []
        for snap in row.get("snapshots", []):
            visual = snap.get("visual")
            audio = snap.get("audio")
            snapshots.append(
                LiveStreamSnapshot(
                    timestamp_sec=float(snap.get("timestamp_sec", 0.0)),
                    transcript_excerpt=str(snap.get("transcript_excerpt", "")),
                    visual=VisualCue(**visual) if isinstance(visual, dict) else None,
                    audio=AudioCue(**audio) if isinstance(audio, dict) else None,
                    concept_hints=tuple(snap.get("concept_hints", ())),
                )
            )
        stream = LiveStreamInfo(
            stream_id=str(row["stream_id"]),
            title=str(row.get("title", row["stream_id"])),
            status=str(row.get("status", "live")),
            livekit_room_name=str(row.get("livekit_room_name", row["stream_id"])),
            livekit_url=str(row.get("livekit_url", os.environ.get("LIVEKIT_URL", "wss://livekit.example"))),
            curriculum_topic=str(row.get("curriculum_topic", "kids-plasma")),
            topic=str(row.get("topic", "")),
            tags=tuple(str(tag) for tag in row.get("tags", ())),
            snapshots=tuple(snapshots),
        )
        streams.append(stream)
    return streams


def join_live_stream(
    base_url: str,
    stream_id: str,
    participant_address: str,
    *,
    role: str = "viewer",
    timeout_sec: int = 10,
) -> tuple[LiveStreamInfo, "LiveKitCredentials"] | None:
    """Join a live stream via SocialServer and return stream info + credentials."""
    from allm.researcher.livekit_types import LiveKitCredentials

    root = base_url.rstrip("/")
    body = json.dumps({"role": role}).encode("utf-8")
    request = urllib.request.Request(
        f"{root}/api/streams/{stream_id}/join",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-user-address": participant_address.lower(),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        logger.warning("social join stream failed for %s: %s", stream_id, exc)
        return None

    row = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(row, dict):
        return None
    stream = _parse_stream_row(row)
    if stream is None:
        return None
    room_info = row.get("recordingRoomInfo") or {}
    token = room_info.get("viewerToken") or room_info.get("publisherToken") or ""
    url = room_info.get("url") or stream.livekit_url
    room_name = room_info.get("roomName") or stream.livekit_room_name
    if not token:
        logger.warning("join stream %s returned no LiveKit token", stream_id)
        return None
    credentials = LiveKitCredentials(
        stream_id=stream.stream_id,
        url=str(url),
        room_name=str(room_name),
        token=str(token),
        identity=participant_address.lower(),
    )
    return stream, credentials


def leave_live_stream(
    base_url: str,
    stream_id: str,
    participant_address: str,
    *,
    timeout_sec: int = 10,
) -> bool:
    """Notify SocialServer that the observer left the stream."""
    root = base_url.rstrip("/")
    request = urllib.request.Request(
        f"{root}/api/streams/{stream_id}/leave",
        data=b"{}",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-user-address": participant_address.lower(),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return bool(payload.get("success"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        logger.warning("social leave stream failed for %s: %s", stream_id, exc)
        return False
