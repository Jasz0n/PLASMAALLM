"""Object identity persistence across multiple workshop sources."""

from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.motion_continuity import (
    _cue_tokens,
    _primary_object,
    _union_find_merge,
    _union_find_parent,
)
from allm.researcher.multimodal_types import SyncedEvidence

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


class ObjectIdentityRecord(BaseModel):
    """Persistent object identity spanning multiple workshop sources."""

    model_config = ConfigDict(frozen=True)

    identity_id: str
    primary_object: str
    source_ids: tuple[str, ...] = ()
    track_ids: tuple[str, ...] = ()
    identity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


class _TrackAggregate(BaseModel):
    """One within-source motion track or standalone cue cluster."""

    model_config = ConfigDict(frozen=True)

    track_key: str
    source_id: str
    primary_object: str
    rows: tuple[SyncedEvidence, ...]
    tokens: frozenset[str]
    concept_hints: frozenset[str]
    motion_vectors: frozenset[str]
    colors: frozenset[str]
    tags: frozenset[str]


def _normalize_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if len(token) >= 3}


def _build_track_aggregates(rows: list[SyncedEvidence]) -> list[_TrackAggregate]:
    grouped: dict[str, list[SyncedEvidence]] = defaultdict(list)
    for row in rows:
        key = row.motion_track_id or f"{row.source_id}-solo-{row.timestamp_sec:.0f}"
        grouped[key].append(row)

    aggregates: list[_TrackAggregate] = []
    for track_key, track_rows in grouped.items():
        ordered = sorted(track_rows, key=lambda row: row.timestamp_sec)
        tokens: set[str] = set()
        hints: set[str] = set()
        vectors: set[str] = set()
        colors: set[str] = set()
        tags: set[str] = set()
        for row in ordered:
            tokens.update(_cue_tokens(row))
            hints.update(hint.lower() for hint in row.concept_hints)
            if row.visual is not None:
                if row.visual.motion_vector:
                    vectors.add(row.visual.motion_vector)
                colors.update(color.lower() for color in row.visual.dominant_colors)
                tags.update(tag.lower().replace("_", "-") for tag in row.visual.tags)
        primary = _primary_object(tokens)
        aggregates.append(
            _TrackAggregate(
                track_key=track_key,
                source_id=ordered[0].source_id,
                primary_object=primary,
                rows=tuple(ordered),
                tokens=frozenset(tokens),
                concept_hints=frozenset(hints),
                motion_vectors=frozenset(vectors),
                colors=frozenset(colors),
                tags=frozenset(tags),
            )
        )
    return aggregates


def cross_source_identity_score(left: _TrackAggregate, right: _TrackAggregate) -> float:
    """Score whether two source tracks depict the same persistent object."""
    if left.source_id == right.source_id:
        return 0.0
    if left.primary_object != right.primary_object:
        return 0.0

    score = 0.28
    union = left.tokens | right.tokens
    if union:
        score += 0.32 * (len(left.tokens & right.tokens) / len(union))

    if left.concept_hints & right.concept_hints:
        score += 0.18

    shared_tags = left.tags & right.tags
    if shared_tags:
        score += min(0.14, 0.05 * len(shared_tags))

    if left.colors & right.colors:
        score += 0.08

    if left.motion_vectors and right.motion_vectors:
        if left.motion_vectors & right.motion_vectors:
            score += 0.12
        elif any(
            (left_vector, right_vector) in _COMPATIBLE_VECTORS
            or (right_vector, left_vector) in _COMPATIBLE_VECTORS
            for left_vector in left.motion_vectors
            for right_vector in right.motion_vectors
        ):
            score += 0.08

    magnet_family = {"magnet", "magnetical", "magnetic", "plasma", "field", "rotation"}
    if (
        left.primary_object in {"magnet", "plasma", "field", "rotation"}
        and right.primary_object in {"magnet", "plasma", "field", "rotation"}
    ):
        left_family = left.tokens & magnet_family
        right_family = right.tokens & magnet_family
        if left_family & right_family:
            score += 0.1

    return round(min(1.0, score), 4)


def _build_identity_summary(
    *,
    identity_id: str,
    aggregates: list[_TrackAggregate],
    identity_score: float,
) -> str:
    source_ids = sorted({aggregate.source_id for aggregate in aggregates})
    primary = aggregates[0].primary_object
    if len(source_ids) == 1:
        return f"Identity {identity_id}: {primary} in {source_ids[0]}"
    return (
        f"Identity {identity_id}: {primary} across {len(source_ids)} workshops "
        f"({', '.join(source_ids)}) score {identity_score:.2f}"
    )


def link_object_identities(
    rows: list[SyncedEvidence],
    *,
    min_score: float = 0.30,
    curriculum_topic: str = "kids-plasma",
) -> tuple[list[SyncedEvidence], tuple[ObjectIdentityRecord, ...]]:
    """Assign persistent object identity ids across workshop sources."""
    if not rows:
        return [], ()

    aggregates = _build_track_aggregates(rows)
    size = len(aggregates)
    parents = list(range(size))
    pair_scores: dict[tuple[int, int], float] = {}

    for left in range(size):
        for right in range(left + 1, size):
            score = cross_source_identity_score(aggregates[left], aggregates[right])
            if score >= min_score:
                pair_scores[(left, right)] = score
                _union_find_merge(parents, left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        components[_union_find_parent(parents, index)].append(index)

    identity_number = 1
    identity_by_track_key: dict[str, ObjectIdentityRecord] = {}
    enriched_rows: list[SyncedEvidence] = []
    records: list[ObjectIdentityRecord] = []

    for indices in components.values():
        component_aggregates = [aggregates[index] for index in indices]
        tokens = set()
        for aggregate in component_aggregates:
            tokens.update(aggregate.tokens)
        primary = _primary_object(tokens)
        identity_id = f"oid-{curriculum_topic}-{primary}-{identity_number}"
        identity_number += 1

        source_ids = tuple(sorted({aggregate.source_id for aggregate in component_aggregates}))
        track_ids = tuple(sorted(aggregate.track_key for aggregate in component_aggregates))
        component_pair_scores = [
            pair_scores[(left, right)]
            for left in indices
            for right in indices
            if left < right and (left, right) in pair_scores
        ]
        identity_score = (
            sum(component_pair_scores) / len(component_pair_scores)
            if component_pair_scores
            else 0.0
        )
        summary = _build_identity_summary(
            identity_id=identity_id,
            aggregates=component_aggregates,
            identity_score=identity_score,
        )
        record = ObjectIdentityRecord(
            identity_id=identity_id,
            primary_object=primary,
            source_ids=source_ids,
            track_ids=track_ids,
            identity_score=round(identity_score, 4),
            summary=summary,
        )
        records.append(record)
        for aggregate in component_aggregates:
            identity_by_track_key[aggregate.track_key] = record

    for row in rows:
        track_key = row.motion_track_id or f"{row.source_id}-solo-{row.timestamp_sec:.0f}"
        record = identity_by_track_key.get(track_key)
        if record is None:
            enriched_rows.append(row)
            continue
        linked_sources = tuple(source for source in record.source_ids if source != row.source_id)
        bonus = 0.0
        if linked_sources:
            bonus += 0.03
        if len(linked_sources) >= 1:
            bonus += 0.02
        confidence = min(1.0, round(row.confidence + bonus, 4))
        enriched_rows.append(
            row.model_copy(
                update={
                    "object_identity_id": record.identity_id,
                    "linked_source_ids": linked_sources,
                    "identity_summary": record.summary if linked_sources else None,
                    "confidence": confidence,
                }
            )
        )

    enriched_rows.sort(key=lambda row: (row.source_id, row.timestamp_sec))
    records.sort(key=lambda record: record.identity_id)
    return enriched_rows, tuple(records)


def enrich_object_identities(
    rows: list[SyncedEvidence],
    *,
    min_score: float = 0.30,
    curriculum_topic: str = "kids-plasma",
) -> tuple[list[SyncedEvidence], tuple[ObjectIdentityRecord, ...]]:
    """Attach cross-workshop object identity metadata to synced evidence rows."""
    return link_object_identities(
        rows,
        min_score=min_score,
        curriculum_topic=curriculum_topic,
    )
