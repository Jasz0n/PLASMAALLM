"""Evidence broker — Researcher packages for debate 'show me' requests."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.multimodal import retrieve_synced_evidence
from allm.researcher.multimodal_types import SyncedEvidence
from allm.researcher.types import KnowledgePackage


class EvidenceHit(BaseModel):
    """One retrieved evidence row with package provenance."""

    model_config = ConfigDict(frozen=True)

    package_id: str
    provider: str
    evidence: SyncedEvidence


class EvidenceBundle(BaseModel):
    """Formatted evidence package for Teacher/debate display."""

    model_config = ConfigDict(frozen=True)

    query: str
    topic: str = ""
    hits: tuple[EvidenceHit, ...] = ()
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


def _format_summary(hits: Sequence[EvidenceHit]) -> str:
    if not hits:
        return "No synchronized visual evidence found."
    lines: list[str] = []
    for hit in hits[:3]:
        row = hit.evidence
        visual = row.visual.description if row.visual else "visual unavailable"
        live = " [LIVE]" if row.is_live else ""
        frames = ""
        if row.visual and row.visual.frame_start is not None:
            frames = f" frames {row.visual.frame_start}-{row.visual.frame_end}"
        caption = f" | {row.visual.caption}" if row.visual and row.visual.caption else ""
        ocr_note = ""
        if row.visual and row.visual.ocr_text:
            ocr_note = f" | ocr: {row.visual.ocr_text[:80]}"
        elif row.visual and row.visual.diagram_labels:
            ocr_note = f" | labels: {', '.join(row.visual.diagram_labels[:4])}"
        analytics_note = ""
        if row.visual and row.visual.analytics_summary:
            analytics_note = f" | analytics: {row.visual.analytics_summary[:80]}"
        elif row.visual and row.visual.visual_features:
            analytics_note = f" | analytics: {', '.join(row.visual.visual_features[:4])}"
        motion_note = ""
        if row.visual and row.visual.motion_summary:
            motion_note = f" | motion: {row.visual.motion_summary[:80]}"
        elif row.visual and row.visual.motion_vector:
            motion_note = f" | motion: {row.visual.motion_vector}"
        continuity_note = ""
        if row.continuity_summary:
            continuity_note = f" | track: {row.continuity_summary[:80]}"
        elif row.motion_track_id and row.linked_cue_timestamps:
            continuity_note = (
                f" | track: {row.motion_track_id} linked "
                f"{len(row.linked_cue_timestamps)} cues"
            )
        identity_note = ""
        if row.identity_summary:
            identity_note = f" | identity: {row.identity_summary[:80]}"
        elif row.object_identity_id and row.linked_source_ids:
            identity_note = (
                f" | identity: {row.object_identity_id} across "
                f"{len(row.linked_source_ids)} workshops"
            )
        audio_note = ""
        if row.audio:
            if row.audio.analysis:
                audio_note = f" | audio: {row.audio.analysis[:80]}"
            elif row.audio.features:
                audio_note = f" | audio: {', '.join(row.audio.features[:4])}"
        lines.append(
            f"{row.source_id} @{row.timestamp_sec:.0f}s{frames}{live}: {visual}{caption}{ocr_note}{analytics_note}{motion_note}{continuity_note}{identity_note}{audio_note} "
            f"(confidence {row.confidence:.2f})"
        )
    return "; ".join(lines)


class EvidenceBroker:
    """Search stored Knowledge Packages for multimodal evidence."""

    def __init__(self, packages: Sequence[KnowledgePackage]) -> None:
        self._packages = tuple(packages)

    def show_me(
        self,
        query: str,
        *,
        topic: str | None = None,
        limit: int = 5,
    ) -> EvidenceBundle:
        """Retrieve evidence hits across all packages matching query/topic."""
        hits: list[EvidenceHit] = []
        for package in self._packages:
            if topic and package.curriculum_topic and package.curriculum_topic != topic:
                continue
            for row in retrieve_synced_evidence(package, query=query, limit=limit):
                hits.append(
                    EvidenceHit(
                        package_id=package.id,
                        provider=package.provider,
                        evidence=row,
                    )
                )
        hits.sort(key=lambda row: -row.evidence.confidence)
        trimmed = tuple(hits[:limit])
        confidence = max((row.evidence.confidence for row in trimmed), default=0.0)
        return EvidenceBundle(
            query=query,
            topic=topic or "",
            hits=trimmed,
            confidence=round(confidence, 4),
            summary=_format_summary(trimmed),
        )

    @classmethod
    def from_store(cls, store) -> "EvidenceBroker":
        """Build broker from a Researcher recommendation queue store."""
        from allm.researcher.queue import RecommendationQueue

        queue = RecommendationQueue(store)
        return cls(queue.packages())
