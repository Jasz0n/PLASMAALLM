"""Cross-cue motion continuity — link visual evidence across a workshop timeline."""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.multimodal_types import SyncedEvidence

_OBJECT_HINTS = (
    "magnet",
    "plasma",
    "field",
    "pole",
    "reactor",
    "coil",
    "gauss",
    "rotation",
)
_COMPATIBLE_VECTORS = frozenset(
    {
        ("rotation", "oscillation"),
        ("oscillation", "rotation"),
        ("oscillation", "translation"),
        ("rotation", "translation"),
        ("translation", "static"),
        ("static", "translation"),
    }
)
_TOKEN = re.compile(r"[a-z0-9]+")


class MotionContinuityTrack(BaseModel):
    """One continuous object/motion thread across multiple timeline cues."""

    model_config = ConfigDict(frozen=True)

    track_id: str
    source_id: str
    timestamps: tuple[float, ...] = ()
    primary_object: str = "object"
    continuity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


def _normalize_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if len(token) >= 3}


def _cue_tokens(row: SyncedEvidence) -> set[str]:
    tokens: set[str] = set()
    for hint in row.concept_hints:
        tokens.update(_normalize_tokens(hint))
    visual = row.visual
    if visual is None:
        return tokens
    for tag in visual.tags:
        tokens.add(tag.lower().replace("_", "-"))
        tokens.update(_normalize_tokens(tag))
    tokens.update(_normalize_tokens(visual.description))
    if visual.caption:
        tokens.update(_normalize_tokens(visual.caption))
    if visual.ocr_text:
        tokens.update(_normalize_tokens(visual.ocr_text))
    for color in visual.dominant_colors:
        tokens.add(color.lower())
    for feature in visual.visual_features:
        tokens.update(_normalize_tokens(feature.replace("_", " ")))
    return tokens


def _primary_object(tokens: set[str]) -> str:
    for hint in _OBJECT_HINTS:
        if any(hint in token for token in tokens):
            return hint
    if tokens:
        return sorted(tokens)[0]
    return "object"


def _union_find_parent(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _union_find_merge(parents: list[int], left: int, right: int) -> None:
    root_left = _union_find_parent(parents, left)
    root_right = _union_find_parent(parents, right)
    if root_left != root_right:
        parents[root_right] = root_left


def continuity_score(left: SyncedEvidence, right: SyncedEvidence) -> float:
    """Score whether two cues belong to the same motion/object track."""
    if left.source_id != right.source_id:
        return 0.0
    if right.timestamp_sec <= left.timestamp_sec:
        return 0.0

    score = 0.0
    tokens_left = _cue_tokens(left)
    tokens_right = _cue_tokens(right)
    union = tokens_left | tokens_right
    if union:
        score += 0.35 * (len(tokens_left & tokens_right) / len(union))

    hints_left = {hint.lower() for hint in left.concept_hints}
    hints_right = {hint.lower() for hint in right.concept_hints}
    if hints_left & hints_right:
        score += 0.2

    time_gap = right.timestamp_sec - left.timestamp_sec
    if time_gap <= 300:
        score += 0.12
    if time_gap <= 120:
        score += 0.08

    visual_left = left.visual
    visual_right = right.visual
    if visual_left is not None and visual_right is not None:
        shared_tags = set(visual_left.tags) & set(visual_right.tags)
        if shared_tags:
            score += min(0.15, 0.05 * len(shared_tags))

        if (
            visual_left.frame_end is not None
            and visual_right.frame_start is not None
        ):
            frame_gap = visual_right.frame_start - visual_left.frame_end
            if 0 <= frame_gap <= 120:
                score += 0.18
            elif 0 <= frame_gap <= 240:
                score += 0.08

        colors_left = set(visual_left.dominant_colors)
        colors_right = set(visual_right.dominant_colors)
        if colors_left & colors_right:
            score += 0.08

        vector_left = visual_left.motion_vector
        vector_right = visual_right.motion_vector
        if vector_left and vector_right:
            if vector_left == vector_right:
                score += 0.12
            elif (vector_left, vector_right) in _COMPATIBLE_VECTORS:
                score += 0.08

    return round(min(1.0, score), 4)


def _build_track_summary(
    *,
    track_id: str,
    rows: list[SyncedEvidence],
    continuity_score: float,
) -> str:
    timestamps = sorted(row.timestamp_sec for row in rows)
    if len(timestamps) == 1:
        return f"Track {track_id} @ {timestamps[0]:.0f}s"
    return (
        f"Track {track_id}: {len(rows)} linked cues "
        f"@{timestamps[0]:.0f}s–@{timestamps[-1]:.0f}s "
        f"(continuity {continuity_score:.2f})"
    )


def link_motion_continuity(
    rows: list[SyncedEvidence],
    *,
    min_score: float = 0.35,
) -> tuple[list[SyncedEvidence], tuple[MotionContinuityTrack, ...]]:
    """Assign motion track ids and linked timestamps across one workshop timeline."""
    if not rows:
        return [], ()

    grouped: dict[str, list[SyncedEvidence]] = defaultdict(list)
    for row in rows:
        grouped[row.source_id].append(row)

    enriched: list[SyncedEvidence] = []
    tracks: list[MotionContinuityTrack] = []

    for source_id, source_rows in grouped.items():
        ordered = sorted(source_rows, key=lambda row: row.timestamp_sec)
        size = len(ordered)
        parents = list(range(size))
        pair_scores: dict[tuple[int, int], float] = {}

        for left in range(size):
            for right in range(left + 1, size):
                score = continuity_score(ordered[left], ordered[right])
                if score >= min_score:
                    pair_scores[(left, right)] = score
                    _union_find_merge(parents, left, right)

        components: dict[int, list[int]] = defaultdict(list)
        for index in range(size):
            components[_union_find_parent(parents, index)].append(index)

        track_number = 1
        for indices in components.values():
            component_rows = [ordered[index] for index in indices]
            tokens = set()
            for row in component_rows:
                tokens.update(_cue_tokens(row))
            primary = _primary_object(tokens)
            track_id = f"{source_id}-{primary}-{track_number}"
            track_number += 1

            timestamps = tuple(sorted(row.timestamp_sec for row in component_rows))
            if len(indices) >= 2:
                component_pair_scores = [
                    pair_scores[(left, right)]
                    for left in indices
                    for right in indices
                    if left < right and (left, right) in pair_scores
                ]
                track_score = (
                    sum(component_pair_scores) / len(component_pair_scores)
                    if component_pair_scores
                    else min_score
                )
            else:
                track_score = 0.0

            summary = _build_track_summary(
                track_id=track_id,
                rows=component_rows,
                continuity_score=track_score,
            )
            tracks.append(
                MotionContinuityTrack(
                    track_id=track_id,
                    source_id=source_id,
                    timestamps=timestamps,
                    primary_object=primary,
                    continuity_score=round(track_score, 4),
                    summary=summary,
                )
            )

            for row in component_rows:
                linked = tuple(ts for ts in timestamps if ts != row.timestamp_sec)
                bonus = 0.0
                if linked:
                    bonus += 0.03
                if len(linked) >= 2:
                    bonus += 0.02
                confidence = min(1.0, round(row.confidence + bonus, 4))
                enriched.append(
                    row.model_copy(
                        update={
                            "motion_track_id": track_id,
                            "linked_cue_timestamps": linked,
                            "continuity_summary": summary if len(component_rows) > 1 else None,
                            "confidence": confidence,
                        }
                    )
                )

    enriched.sort(key=lambda row: (row.source_id, row.timestamp_sec))
    tracks.sort(key=lambda track: (track.source_id, track.timestamps[0] if track.timestamps else 0.0))
    return enriched, tuple(tracks)


def enrich_synced_evidence_continuity(
    rows: list[SyncedEvidence],
    *,
    min_score: float = 0.35,
) -> tuple[list[SyncedEvidence], tuple[MotionContinuityTrack, ...]]:
    """Attach cross-cue motion continuity metadata to synced evidence rows."""
    return link_motion_continuity(rows, min_score=min_score)
