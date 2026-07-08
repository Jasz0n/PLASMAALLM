"""Vision captioning for extracted frames — stub offline, Ollama when available."""

from __future__ import annotations

import os
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.researcher.ollama_vision import (
    DEFAULT_VISION_MODEL,
    OllamaVisionClient,
    build_vision_prompt,
    ollama_reachable,
)
from typing import Protocol, runtime_checkable

logger = get_logger("researcher.vision_caption")


@runtime_checkable
class VisionCaptioner(Protocol):
    """Generate a human-readable caption for one visual moment."""

    def caption(
        self,
        *,
        transcript_excerpt: str,
        tags: tuple[str, ...] = (),
        concept_hints: tuple[str, ...] = (),
        frame_path: str | None = None,
    ) -> str: ...


class StubVisionCaptioner:
    """Offline captioner using transcript context and tags (no GPU)."""

    def caption(
        self,
        *,
        transcript_excerpt: str,
        tags: tuple[str, ...] = (),
        concept_hints: tuple[str, ...] = (),
        frame_path: str | None = None,
    ) -> str:
        excerpt = transcript_excerpt.strip()[:140]
        tag_text = ", ".join(tags[:4])
        hint_text = ", ".join(concept_hints[:3])
        prefix = "Frame captured" if frame_path else "Transcript-aligned scene"
        parts = [prefix, excerpt]
        if tag_text:
            parts.append(f"tags: {tag_text}")
        if hint_text:
            parts.append(f"concepts: {hint_text}")
        return " — ".join(part for part in parts if part)


class OllamaVisionCaptioner:
    """Caption extracted frames via Ollama vision models; stub fallback otherwise."""

    def __init__(
        self,
        client: OllamaVisionClient,
        fallback: VisionCaptioner | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or StubVisionCaptioner()

    def caption(
        self,
        *,
        transcript_excerpt: str,
        tags: tuple[str, ...] = (),
        concept_hints: tuple[str, ...] = (),
        frame_path: str | None = None,
    ) -> str:
        path = Path(frame_path) if frame_path else None
        if path is not None and path.is_file():
            prompt = build_vision_prompt(
                transcript_excerpt=transcript_excerpt,
                tags=tags,
                concept_hints=concept_hints,
            )
            try:
                caption = self._client.describe_image(path, prompt)
                return f"Vision: {caption.strip()}"
            except (OSError, RuntimeError) as exc:
                logger.warning("ollama vision caption failed: %s", exc)
        return self._fallback.caption(
            transcript_excerpt=transcript_excerpt,
            tags=tags,
            concept_hints=concept_hints,
            frame_path=frame_path,
        )


def get_vision_captioner(
    backend: str = "stub",
    *,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
) -> VisionCaptioner:
    """Resolve caption backend: stub, ollama, or hybrid (ollama with stub fallback)."""
    normalized = backend.lower().strip()
    if normalized in {"stub", "metadata", "offline"}:
        return StubVisionCaptioner()

    model = ollama_model or os.environ.get("ALLM_VISION_MODEL", DEFAULT_VISION_MODEL)
    if normalized in {"ollama", "hybrid", "auto"}:
        client = OllamaVisionClient(
            model_id=model,
            base_url=ollama_base_url,
        )
        if normalized == "auto" and not ollama_reachable(client._base_url):
            logger.info("ollama unreachable — using stub vision captioner")
            return StubVisionCaptioner()
        return OllamaVisionCaptioner(client)

    raise ValueError(f"unsupported vision caption backend: {backend}")


def enrich_visual_cue(
    visual: VisualCue,
    *,
    captioner: VisionCaptioner,
    transcript_excerpt: str,
    concept_hints: tuple[str, ...] = (),
    frame_path: str | None = None,
) -> VisualCue:
    """Return visual cue with caption and optional frame path."""
    path = frame_path or visual.frame_path
    caption = captioner.caption(
        transcript_excerpt=transcript_excerpt,
        tags=visual.tags,
        concept_hints=concept_hints,
        frame_path=path,
    )
    description = visual.description
    if caption and caption not in description:
        description = f"{description} | {caption}"
    return visual.model_copy(
        update={
            "frame_path": path,
            "caption": caption,
            "description": description,
        }
    )


def enrich_synced_evidence(
    row: SyncedEvidence,
    *,
    captioner: VisionCaptioner,
    frame_path: str | None = None,
) -> SyncedEvidence:
    """Attach caption and frame path to one synced evidence row."""
    if row.visual is None:
        return row
    visual = enrich_visual_cue(
        row.visual,
        captioner=captioner,
        transcript_excerpt=row.transcript_excerpt,
        concept_hints=row.concept_hints,
        frame_path=frame_path,
    )
    bonus = 0.08 if visual.caption and visual.caption.startswith("Vision:") else 0.05
    confidence = min(1.0, round(row.confidence + (bonus if visual.caption else 0.0), 4))
    return row.model_copy(update={"visual": visual, "confidence": confidence})
