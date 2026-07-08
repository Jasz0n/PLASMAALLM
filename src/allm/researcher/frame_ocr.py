"""OCR on extracted workshop frames — stub metadata, tesseract, or Ollama vision."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from allm.core.logging import get_logger
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.researcher.ollama_vision import (
    DEFAULT_VISION_MODEL,
    OllamaVisionClient,
    ollama_reachable,
)

logger = get_logger("researcher.frame_ocr")

_DIAGRAM_TAG_HINTS = frozenset(
    {
        "diagram",
        "chart",
        "label",
        "blue-plasma",
        "field-beat",
        "magnet-rotation",
        "repulsion",
        "similar-poles",
    }
)
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9+-]+")


@runtime_checkable
class FrameOcr(Protocol):
    """Extract text and diagram labels from one workshop frame."""

    def read_frame(
        self,
        *,
        frame_path: str | None,
        description: str,
        tags: tuple[str, ...] = (),
        concept_hints: tuple[str, ...] = (),
        transcript_excerpt: str = "",
    ) -> tuple[str, tuple[str, ...]]: ...


def tesseract_available() -> bool:
    """Return True when the tesseract CLI is on PATH."""
    return shutil.which("tesseract") is not None


def build_ocr_prompt(
    *,
    transcript_excerpt: str,
    tags: tuple[str, ...] = (),
    concept_hints: tuple[str, ...] = (),
) -> str:
    """Prompt for reading text on workshop diagrams and whiteboards."""
    tag_text = ", ".join(tags[:6])
    hint_text = ", ".join(concept_hints[:4])
    return (
        "Read all visible text, labels, numbers, and short annotations in this workshop "
        "diagram or demonstration frame. Return only the text you can read, one item per line. "
        "If there is no readable text, reply with: no text visible.\n"
        f"Transcript context: {transcript_excerpt[:200]}\n"
        f"Tags: {tag_text or 'none'}\n"
        f"Concepts: {hint_text or 'none'}"
    )


def _labels_from_text(text: str) -> tuple[str, ...]:
    labels: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip(" -•\t")
        if not cleaned or cleaned.lower() == "no text visible":
            continue
        labels.append(cleaned[:80])
    return tuple(dict.fromkeys(labels))


def _infer_stub_labels(
    description: str,
    tags: tuple[str, ...],
    concept_hints: tuple[str, ...],
) -> tuple[str, ...]:
    labels: list[str] = []
    for tag in tags:
        token = tag.replace("-", " ").strip()
        if token:
            labels.append(token)
    for hint in concept_hints:
        hint = hint.strip()
        if hint:
            labels.append(hint)
    if tags and (set(tags) & _DIAGRAM_TAG_HINTS or len(tags) >= 2):
        labels.append("diagram")
    for token in _TOKEN_SPLIT.split(description):
        if len(token) >= 4 and token.lower() not in {"visible", "between", "close"}:
            labels.append(token.lower())
    return tuple(dict.fromkeys(label for label in labels if label))


class StubFrameOcr:
    """Offline OCR using fixture metadata when frames are unavailable."""

    def read_frame(
        self,
        *,
        frame_path: str | None,
        description: str,
        tags: tuple[str, ...] = (),
        concept_hints: tuple[str, ...] = (),
        transcript_excerpt: str = "",
    ) -> tuple[str, tuple[str, ...]]:
        labels = _infer_stub_labels(description, tags, concept_hints)
        if not labels:
            return "", ()
        prefix = "Frame OCR" if frame_path else "Metadata OCR"
        text = f"{prefix}: {', '.join(labels[:8])}"
        return text, labels


class TesseractFrameOcr:
    """Run tesseract on extracted frames; fall back to stub metadata."""

    def __init__(self, fallback: FrameOcr | None = None) -> None:
        self._fallback = fallback or StubFrameOcr()

    def read_frame(
        self,
        *,
        frame_path: str | None,
        description: str,
        tags: tuple[str, ...] = (),
        concept_hints: tuple[str, ...] = (),
        transcript_excerpt: str = "",
    ) -> tuple[str, tuple[str, ...]]:
        path = Path(frame_path) if frame_path else None
        if path is not None and path.is_file() and tesseract_available():
            try:
                result = subprocess.run(
                    ["tesseract", str(path), "stdout", "-l", "eng"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    raw = (result.stdout or "").strip()
                    labels = _labels_from_text(raw)
                    if labels:
                        return raw, labels
            except (subprocess.SubprocessError, OSError) as exc:
                logger.warning("tesseract OCR failed: %s", exc)
        return self._fallback.read_frame(
            frame_path=frame_path,
            description=description,
            tags=tags,
            concept_hints=concept_hints,
            transcript_excerpt=transcript_excerpt,
        )


class OllamaFrameOcr:
    """Read diagram text via Ollama vision; fall back to tesseract or stub."""

    def __init__(
        self,
        client: OllamaVisionClient,
        fallback: FrameOcr | None = None,
    ) -> None:
        self._client = client
        self._fallback = fallback or TesseractFrameOcr(StubFrameOcr())

    def read_frame(
        self,
        *,
        frame_path: str | None,
        description: str,
        tags: tuple[str, ...] = (),
        concept_hints: tuple[str, ...] = (),
        transcript_excerpt: str = "",
    ) -> tuple[str, tuple[str, ...]]:
        path = Path(frame_path) if frame_path else None
        if path is not None and path.is_file():
            prompt = build_ocr_prompt(
                transcript_excerpt=transcript_excerpt,
                tags=tags,
                concept_hints=concept_hints,
            )
            try:
                raw = self._client.describe_image(path, prompt).strip()
                labels = _labels_from_text(raw)
                if labels:
                    return f"OCR: {raw}", labels
            except (OSError, RuntimeError) as exc:
                logger.warning("ollama OCR failed: %s", exc)
        return self._fallback.read_frame(
            frame_path=frame_path,
            description=description,
            tags=tags,
            concept_hints=concept_hints,
            transcript_excerpt=transcript_excerpt,
        )


def get_frame_ocr(
    backend: str = "auto",
    *,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
) -> FrameOcr:
    """Resolve OCR backend: stub, tesseract, ollama, or auto."""
    normalized = backend.lower().strip()
    stub = StubFrameOcr()
    if normalized in {"stub", "metadata", "offline"}:
        return stub

    if normalized == "tesseract":
        return TesseractFrameOcr(stub)

    model = ollama_model or os.environ.get("ALLM_OCR_MODEL") or os.environ.get(
        "ALLM_VISION_MODEL", DEFAULT_VISION_MODEL
    )
    if normalized == "ollama":
        return OllamaFrameOcr(OllamaVisionClient(model_id=model, base_url=ollama_base_url), stub)

    if normalized in {"auto", "hybrid"}:
        if tesseract_available():
            return TesseractFrameOcr(stub)
        client = OllamaVisionClient(model_id=model, base_url=ollama_base_url)
        if ollama_reachable(client._base_url):
            return OllamaFrameOcr(client, TesseractFrameOcr(stub))
        logger.info("no tesseract or ollama — using stub frame OCR")
        return stub

    raise ValueError(f"unsupported OCR backend: {backend}")


def enrich_visual_cue_ocr(
    visual: VisualCue,
    *,
    ocr: FrameOcr,
    transcript_excerpt: str = "",
    concept_hints: tuple[str, ...] = (),
) -> VisualCue:
    """Return visual cue with OCR text and diagram labels."""
    text, labels = ocr.read_frame(
        frame_path=visual.frame_path,
        description=visual.description,
        tags=visual.tags,
        concept_hints=concept_hints,
        transcript_excerpt=transcript_excerpt,
    )
    description = visual.description
    if text and text not in description:
        description = f"{description} | {text}"
    return visual.model_copy(
        update={
            "ocr_text": text or None,
            "diagram_labels": labels,
            "description": description,
        }
    )


def enrich_synced_evidence_ocr(
    row: SyncedEvidence,
    *,
    ocr: FrameOcr,
) -> SyncedEvidence:
    """Attach OCR text and diagram labels to one synced evidence row."""
    if row.visual is None:
        return row
    visual = enrich_visual_cue_ocr(
        row.visual,
        ocr=ocr,
        transcript_excerpt=row.transcript_excerpt,
        concept_hints=row.concept_hints,
    )
    bonus = 0.0
    if visual.ocr_text:
        bonus += 0.04
    if visual.diagram_labels:
        bonus += 0.03
    if visual.ocr_text and visual.ocr_text.startswith("OCR:"):
        bonus += 0.02
    confidence = min(1.0, round(row.confidence + bonus, 4))
    return row.model_copy(update={"visual": visual, "confidence": confidence})
