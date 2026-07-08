"""LiveKit value objects for Researcher live stream observation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.multimodal_types import AudioCue, VisualCue


class LiveKitConfig(BaseModel):
    """LiveKit server credentials (same env vars as SocialServer)."""

    model_config = ConfigDict(frozen=True)

    url: str
    api_key: str
    api_secret: str


class LiveKitCredentials(BaseModel):
    """Connection bundle for a Researcher observer joining a room."""

    model_config = ConfigDict(frozen=True)

    stream_id: str
    url: str
    room_name: str
    token: str
    identity: str


class LiveStreamSnapshot(BaseModel):
    """One offline or captured live moment."""

    model_config = ConfigDict(frozen=True)

    timestamp_sec: float = Field(ge=0.0)
    transcript_excerpt: str = ""
    visual: VisualCue | None = None
    audio: AudioCue | None = None
    concept_hints: tuple[str, ...] = ()


class LiveStreamInfo(BaseModel):
    """A live or recently live workshop stream."""

    model_config = ConfigDict(frozen=True)

    stream_id: str
    title: str
    status: str = "live"
    livekit_room_name: str
    livekit_url: str
    curriculum_topic: str = "kids-plasma"
    topic: str = ""
    tags: tuple[str, ...] = ()
    snapshots: tuple[LiveStreamSnapshot, ...] = ()


class LiveFrameCapture(BaseModel):
    """One frame saved from a LiveKit video track."""

    model_config = ConfigDict(frozen=True)

    stream_id: str
    timestamp_sec: float = Field(ge=0.0)
    frame_path: str
    participant_identity: str = ""


class LiveMediaCapture(BaseModel):
    """Combined video frame and audio clip from one live moment."""

    model_config = ConfigDict(frozen=True)

    stream_id: str
    timestamp_sec: float = Field(ge=0.0)
    frame_path: str | None = None
    audio_clip_path: str | None = None
    participant_identity: str = ""
