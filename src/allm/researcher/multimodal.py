"""Multimodal evidence types and transcript synchronization."""

from __future__ import annotations

import json
import re
from pathlib import Path

from allm.researcher.multimodal_types import (
    AudioCue,
    SyncedEvidence,
    TimelineCue,
    VideoTimelineFixture,
    VisualCue,
)
from allm.researcher.types import KnowledgePackage

__all__ = [
    "AudioCue",
    "SyncedEvidence",
    "TimelineCue",
    "VideoTimelineFixture",
    "VisualCue",
    "attach_multimodal_evidence",
    "discover_video_fixtures",
    "load_video_fixture",
    "retrieve_synced_evidence",
    "sync_fixtures_with_workshop_dir",
    "sync_transcript_cues",
    "unsynced_video_gap",
    "count_video_mentions",
    "VIDEO_MENTION_PHRASES",
]


def load_video_fixture(path: Path | str) -> VideoTimelineFixture:
    """Load one JSON video timeline fixture."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return VideoTimelineFixture.model_validate(data)


def discover_video_fixtures(directory: Path | str) -> list[VideoTimelineFixture]:
    """Load all timeline fixtures in a directory."""
    root = Path(directory)
    if not root.is_dir():
        return []
    fixtures: list[VideoTimelineFixture] = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("livekit_"):
            continue
        fixtures.append(load_video_fixture(path))
    return fixtures


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[-_]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned.strip())


VIDEO_MENTION_PHRASES: tuple[str, ...] = (
    "in this video",
    "in the video",
    "you've seen in the video",
    "we've seen in this video",
    "show in that video",
    "what you show in that video",
)


def count_video_mentions(text: str) -> int:
    """Count how often a transcript references on-screen video."""
    normalized = _normalize(text)
    return sum(normalized.count(_normalize(phrase)) for phrase in VIDEO_MENTION_PHRASES)


def count_video_mentions_in_dir(workshop_dir: Path | str) -> int:
    """Aggregate video references across workshop transcripts."""
    directory = Path(workshop_dir)
    if not directory.is_dir():
        return 0
    total = 0
    for path in sorted(directory.glob("*.txt")):
        total += count_video_mentions(path.read_text(encoding="utf-8", errors="replace"))
    return total


def unsynced_video_gap(
    workshop_dir: Path | str,
    fixture_dir: Path | str | None,
) -> tuple[int, int]:
    """Return (total_mentions, unsynced_count) for curiosity/planning."""
    mentions = count_video_mentions_in_dir(workshop_dir)
    if fixture_dir is None:
        return mentions, mentions
    fixtures = discover_video_fixtures(fixture_dir)
    synced = sync_fixtures_with_workshop_dir(fixtures, workshop_dir)
    unsynced = max(0, mentions - len(synced))
    return mentions, unsynced


def _phrase_in_transcript(phrase: str, transcript: str) -> bool:
    """Case-insensitive substring match with normalized whitespace."""
    return _normalize(phrase) in _normalize(transcript)


def _excerpt_around_phrase(transcript: str, phrase: str, *, window: int = 120) -> str:
    """Return a short excerpt centered on the matched phrase."""
    normalized = _normalize(transcript)
    needle = _normalize(phrase)
    index = normalized.find(needle)
    if index < 0:
        return phrase
    start = max(0, index - window // 2)
    end = min(len(normalized), index + len(needle) + window // 2)
    return normalized[start:end].strip()


def sync_transcript_cues(
    fixture: VideoTimelineFixture,
    transcript_text: str,
) -> list[SyncedEvidence]:
    """Align fixture cues to transcript phrases that appear in the text."""
    synced: list[SyncedEvidence] = []
    for cue in fixture.cues:
        if not _phrase_in_transcript(cue.transcript_phrase, transcript_text):
            continue
        confidence = 0.87 if cue.visual is not None else 0.65
        synced.append(
            SyncedEvidence(
                source_id=fixture.source_id,
                timestamp_sec=cue.timestamp_sec,
                transcript_excerpt=_excerpt_around_phrase(transcript_text, cue.transcript_phrase),
                visual=cue.visual,
                audio=cue.audio,
                concept_hints=cue.concept_hints,
                confidence=confidence,
            )
        )
    return synced


def sync_fixtures_with_workshop_dir(
    fixtures: list[VideoTimelineFixture],
    workshop_dir: Path | str,
) -> list[SyncedEvidence]:
    """Match fixtures to workshop transcripts by filename."""
    directory = Path(workshop_dir)
    all_synced: list[SyncedEvidence] = []
    for fixture in fixtures:
        transcript_path = directory / fixture.transcript_ref
        if not transcript_path.is_file():
            candidates = list(directory.glob(f"*{fixture.transcript_ref}"))
            if not candidates:
                continue
            transcript_path = candidates[0]
        text = transcript_path.read_text(encoding="utf-8", errors="replace")
        all_synced.extend(sync_transcript_cues(fixture, text))
    return all_synced


def attach_multimodal_evidence(
    package: KnowledgePackage,
    evidence: list[SyncedEvidence],
    *,
    curriculum_topic: str | None = None,
) -> KnowledgePackage:
    """Return package with synced multimodal evidence attached."""
    if not evidence:
        return package
    topic = curriculum_topic or package.curriculum_topic
    if topic and package.curriculum_topic and topic != package.curriculum_topic:
        return package
    merged = tuple(dict.fromkeys(package.multimodal_evidence + tuple(evidence)))
    return package.model_copy(update={"multimodal_evidence": merged})


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).split() if len(token) > 2}


def _token_overlap_score(needle: str, haystack: str) -> float:
    """Score query against text using token overlap (supports multi-word queries)."""
    tokens = _tokens(needle)
    if not tokens:
        return 0.0
    normalized = _normalize(haystack)
    if needle in normalized:
        return 1.0
    hits = sum(1 for token in tokens if token in normalized)
    return hits / len(tokens)


def retrieve_synced_evidence(
    package: KnowledgePackage,
    *,
    query: str,
    limit: int = 5,
) -> list[SyncedEvidence]:
    """Retrieve evidence for debate 'show me' — keyword match on tags and excerpts."""
    needle = _normalize(query)
    if not needle:
        return list(package.multimodal_evidence[:limit])
    scored: list[tuple[float, SyncedEvidence]] = []
    for row in package.multimodal_evidence:
        score = _token_overlap_score(needle, row.transcript_excerpt) * 0.5
        if row.visual is not None:
            score += _token_overlap_score(needle, row.visual.description) * 0.4
            score += sum(
                0.2 * _token_overlap_score(needle, tag) for tag in row.visual.tags
            )
            if row.visual.ocr_text:
                score += _token_overlap_score(needle, row.visual.ocr_text) * 0.25
            score += sum(
                0.15 * _token_overlap_score(needle, label) for label in row.visual.diagram_labels
            )
            score += sum(
                0.15 * _token_overlap_score(needle, color) for color in row.visual.dominant_colors
            )
            score += sum(
                0.12 * _token_overlap_score(needle, feature) for feature in row.visual.visual_features
            )
            if row.visual.motion_level and _token_overlap_score(needle, row.visual.motion_level) > 0:
                score += 0.15
            if row.visual.motion_vector and _token_overlap_score(needle, row.visual.motion_vector) > 0:
                score += 0.18
            if row.visual.motion_summary:
                score += _token_overlap_score(needle, row.visual.motion_summary) * 0.12
            if row.visual.motion_score is not None and row.visual.motion_score >= 0.5:
                if _token_overlap_score(needle, "motion") > 0 or _token_overlap_score(needle, "moving") > 0:
                    score += 0.12
            if row.visual.is_diagram and _token_overlap_score(needle, "diagram") > 0:
                score += 0.2
            if row.visual.analytics_summary:
                score += _token_overlap_score(needle, row.visual.analytics_summary) * 0.12
        if row.audio is not None:
            score += _token_overlap_score(needle, row.audio.description) * 0.2
            score += sum(
                0.15 * _token_overlap_score(needle, feature) for feature in row.audio.features
            )
            if row.audio.analysis:
                score += _token_overlap_score(needle, row.audio.analysis) * 0.15
        if row.is_live:
            score += 0.1
        if row.motion_track_id and _token_overlap_score(needle, row.motion_track_id) > 0:
            score += 0.15
        if row.continuity_summary:
            score += _token_overlap_score(needle, row.continuity_summary) * 0.12
        if row.linked_cue_timestamps and _token_overlap_score(needle, "track") > 0:
            score += 0.08
        if row.object_identity_id and _token_overlap_score(needle, row.object_identity_id) > 0:
            score += 0.16
        if row.identity_summary:
            score += _token_overlap_score(needle, row.identity_summary) * 0.12
        if row.linked_source_ids and _token_overlap_score(needle, "workshop") > 0:
            score += 0.08
        if row.live_stream_id and _token_overlap_score(needle, row.live_stream_id) > 0:
            score += 0.2
        for hint in row.concept_hints:
            score += _token_overlap_score(needle, hint) * 0.3
        if score > 0:
            scored.append((score * row.confidence, row))
    scored.sort(key=lambda pair: -pair[0])
    return [row for _, row in scored[:limit]]
