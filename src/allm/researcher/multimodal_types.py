"""Multimodal evidence value objects (no logic — avoids import cycles)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VisualCue(BaseModel):
    """One visual observation linked to a timeline moment."""

    model_config = ConfigDict(frozen=True)

    description: str
    frame_start: int | None = None
    frame_end: int | None = None
    frame_path: str | None = None
    caption: str | None = None
    ocr_text: str | None = None
    diagram_labels: tuple[str, ...] = ()
    motion_level: str | None = None
    motion_vector: str | None = None
    motion_score: float | None = None
    frame_sequence_paths: tuple[str, ...] = ()
    motion_summary: str | None = None
    dominant_colors: tuple[str, ...] = ()
    is_diagram: bool = False
    visual_features: tuple[str, ...] = ()
    analytics_summary: str | None = None
    tags: tuple[str, ...] = ()


class AudioCue(BaseModel):
    """One audio observation linked to a timeline moment."""

    model_config = ConfigDict(frozen=True)

    description: str
    tags: tuple[str, ...] = ()
    clip_path: str | None = None
    features: tuple[str, ...] = ()
    analysis: str | None = None


class TimelineCue(BaseModel):
    """Raw cue from a video timeline fixture."""

    model_config = ConfigDict(frozen=True)

    timestamp_sec: float = Field(ge=0.0)
    transcript_phrase: str
    visual: VisualCue | None = None
    audio: AudioCue | None = None
    concept_hints: tuple[str, ...] = ()


class VideoTimelineFixture(BaseModel):
    """Offline stand-in for decoded video + metadata."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    title: str
    duration_sec: float = Field(ge=0.0)
    transcript_ref: str
    curriculum_topic: str = "kids-plasma"
    cues: tuple[TimelineCue, ...] = ()


class SyncedEvidence(BaseModel):
    """Transcript phrase aligned with visual/audio cues."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    timestamp_sec: float = Field(ge=0.0)
    transcript_excerpt: str
    visual: VisualCue | None = None
    audio: AudioCue | None = None
    concept_hints: tuple[str, ...] = ()
    live_stream_id: str | None = None
    is_live: bool = False
    motion_track_id: str | None = None
    linked_cue_timestamps: tuple[float, ...] = ()
    continuity_summary: str | None = None
    object_identity_id: str | None = None
    linked_source_ids: tuple[str, ...] = ()
    identity_summary: str | None = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class DistilledVisualBrief(BaseModel):
    """Teacher-facing distilled visuals — no raw video streams for students."""

    model_config = ConfigDict(frozen=True)

    brief_id: str
    concept_name: str
    concept_description: str = ""
    images: tuple[str, ...] = ()
    diagram_summary: str | None = None
    explanations: tuple[str, ...] = ()
    experiment_prompt: str | None = None
    questions: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    source_kind: Literal["workshop", "book", ""] = ""
    evidence_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    teacher_notes: str = ""


class TeacherVisualApproval(BaseModel):
    """Teacher approval record for one distilled visual brief."""

    model_config = ConfigDict(frozen=True)

    brief_id: str
    approved: bool = True
    max_images: int = Field(default=2, ge=0, le=6)
    max_questions: int = Field(default=3, ge=0, le=10)
    include_diagram: bool = True
    include_experiment: bool = True
    approved_by: str = "teacher"
    review_note: str = ""


class StudentVisualPackage(BaseModel):
    """Teacher-approved visual assets safe for student delivery."""

    model_config = ConfigDict(frozen=True)

    export_id: str
    concept_name: str
    concept_description: str = ""
    images: tuple[str, ...] = ()
    diagram: str | None = None
    explanations: tuple[str, ...] = ()
    experiment: str | None = None
    questions: tuple[str, ...] = ()
    curriculum_topic: str = ""
    approved_by: str = "teacher"
