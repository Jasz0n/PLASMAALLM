"""Visual analytics for synced workshop evidence — motion, color, diagram detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue

logger = get_logger("researcher.vision_analytics")

_MOTION_HIGH = frozenset(
    {
        "rotating",
        "rotation",
        "chasing",
        "chase",
        "motion",
        "moving",
        "spin",
        "spinning",
        "beat",
        "plasma-motion",
        "magnet-rotation",
    }
)
_MOTION_MODERATE = frozenset(
    {
        "twist",
        "twisting",
        "interaction",
        "repulsion",
        "field-beat",
        "visible",
    }
)
_COLOR_WORDS = (
    "blue",
    "red",
    "green",
    "yellow",
    "orange",
    "purple",
    "violet",
    "white",
    "black",
    "cyan",
    "magenta",
    "pink",
    "gold",
    "silver",
)
_DIAGRAM_HINTS = frozenset(
    {
        "diagram",
        "schematic",
        "graph",
        "chart",
        "whiteboard",
        "sketch",
        "label",
        "axes",
        "plot",
    }
)
_COLOR_TAG = re.compile(r"^([a-z]+)-(?:plasma|field|glow|region|light)")


class VisionAnalytics(BaseModel):
    """Structured visual observations for one timeline moment."""

    model_config = ConfigDict(frozen=True)

    motion_level: str | None = None
    dominant_colors: tuple[str, ...] = ()
    is_diagram: bool = False
    visual_features: tuple[str, ...] = ()
    analytics_summary: str = ""


@runtime_checkable
class VisionAnalyzer(Protocol):
    """Derive motion, color, and diagram signals for one visual moment."""

    def analyze(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        caption: str | None = None,
        ocr_text: str | None = None,
        diagram_labels: tuple[str, ...] = (),
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_path: str | None = None,
    ) -> VisionAnalytics: ...


def _collect_text(
    *,
    description: str,
    tags: tuple[str, ...],
    transcript_excerpt: str,
    caption: str | None,
    ocr_text: str | None,
) -> str:
    parts = [description, transcript_excerpt, caption or "", ocr_text or "", " ".join(tags)]
    return " ".join(part.strip() for part in parts if part).lower()


def _infer_motion(
    text: str,
    tags: tuple[str, ...],
    frame_start: int | None,
    frame_end: int | None,
) -> str | None:
    normalized_tags = {tag.lower().replace("_", "-") for tag in tags}
    if normalized_tags & _MOTION_HIGH or any(word in text for word in _MOTION_HIGH):
        return "high"
    if normalized_tags & _MOTION_MODERATE or any(word in text for word in _MOTION_MODERATE):
        return "moderate"
    if frame_start is not None and frame_end is not None and (frame_end - frame_start) >= 40:
        return "moderate"
    if any(token in text for token in ("static", "still", "close-up", "close up")):
        return "static"
    return None


def _infer_colors(text: str, tags: tuple[str, ...]) -> tuple[str, ...]:
    colors: list[str] = []
    for color in _COLOR_WORDS:
        if color in text:
            colors.append(color)
    for tag in tags:
        token = tag.lower().replace("_", "-")
        match = _COLOR_TAG.match(token)
        if match:
            colors.append(match.group(1))
        for color in _COLOR_WORDS:
            if color in token and color not in colors:
                colors.append(color)
    return tuple(dict.fromkeys(colors))


def _infer_diagram(
    text: str,
    tags: tuple[str, ...],
    ocr_text: str | None,
    diagram_labels: tuple[str, ...],
) -> bool:
    if diagram_labels:
        return True
    if ocr_text and len(ocr_text.strip()) >= 8:
        return True
    normalized_tags = {tag.lower().replace("_", "-") for tag in tags}
    if normalized_tags & _DIAGRAM_HINTS:
        return True
    return any(hint in text for hint in _DIAGRAM_HINTS)


def _build_features(
    *,
    motion_level: str | None,
    dominant_colors: tuple[str, ...],
    is_diagram: bool,
    tags: tuple[str, ...],
) -> tuple[str, ...]:
    features: list[str] = []
    if motion_level:
        features.append(f"motion_{motion_level}")
    for color in dominant_colors:
        features.append(f"color_{color}")
    if is_diagram:
        features.append("diagram")
    for tag in tags[:4]:
        token = tag.lower().replace(" ", "-")
        if token not in features:
            features.append(token)
    return tuple(dict.fromkeys(features))


def _opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401

        return True
    except ImportError:
        return False


def _opencv_dominant_colors(frame_path: Path, limit: int = 3) -> tuple[str, ...]:
    import cv2
    import numpy as np

    image = cv2.imread(str(frame_path))
    if image is None:
        return ()
    small = cv2.resize(image, (64, 64))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(float)
    sat = hsv[:, :, 1].astype(float)
    val = hsv[:, :, 2].astype(float)
    mask = (sat > 40) & (val > 40)
    if not mask.any():
        return ("gray",)
    active_hue = hue[mask]
    buckets = {
        "red": ((0, 15), (165, 180)),
        "orange": ((15, 25),),
        "yellow": ((25, 35),),
        "green": ((35, 85),),
        "cyan": ((85, 100),),
        "blue": ((100, 130),),
        "purple": ((130, 165),),
    }
    scores: dict[str, float] = {}
    for name, ranges in buckets.items():
        total = 0.0
        for low, high in ranges:
            total += float(((active_hue >= low) & (active_hue < high)).sum())
        if total > 0:
            scores[name] = total
    ranked = sorted(scores, key=scores.get, reverse=True)
    return tuple(ranked[:limit])


def _opencv_edge_density(frame_path: Path) -> float:
    import cv2

    image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    small = cv2.resize(image, (128, 128))
    edges = cv2.Canny(small, 80, 160)
    return float((edges > 0).sum()) / float(edges.size)


class StubVisionAnalyzer:
    """Offline analyzer using fixture metadata, captions, and OCR context."""

    def analyze(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        caption: str | None = None,
        ocr_text: str | None = None,
        diagram_labels: tuple[str, ...] = (),
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_path: str | None = None,
    ) -> VisionAnalytics:
        text = _collect_text(
            description=description,
            tags=tags,
            transcript_excerpt=transcript_excerpt,
            caption=caption,
            ocr_text=ocr_text,
        )
        motion_level = _infer_motion(text, tags, frame_start, frame_end)
        dominant_colors = _infer_colors(text, tags)
        is_diagram = _infer_diagram(text, tags, ocr_text, diagram_labels)
        visual_features = _build_features(
            motion_level=motion_level,
            dominant_colors=dominant_colors,
            is_diagram=is_diagram,
            tags=tags,
        )
        parts = ["Transcript-aligned visual"]
        if frame_path:
            parts[0] = "Frame analyzed"
        if motion_level:
            parts.append(f"motion: {motion_level}")
        if dominant_colors:
            parts.append(f"colors: {', '.join(dominant_colors[:4])}")
        if is_diagram:
            parts.append("diagram detected")
        if visual_features:
            parts.append(f"features: {', '.join(visual_features[:6])}")
        return VisionAnalytics(
            motion_level=motion_level,
            dominant_colors=dominant_colors,
            is_diagram=is_diagram,
            visual_features=visual_features,
            analytics_summary=" — ".join(parts),
        )


class OpenCvVisionAnalyzer:
    """Merge metadata heuristics with OpenCV color/edge probes when a frame exists."""

    def __init__(self, fallback: VisionAnalyzer | None = None) -> None:
        self._fallback = fallback or StubVisionAnalyzer()

    def analyze(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        caption: str | None = None,
        ocr_text: str | None = None,
        diagram_labels: tuple[str, ...] = (),
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_path: str | None = None,
    ) -> VisionAnalytics:
        base = self._fallback.analyze(
            description=description,
            tags=tags,
            transcript_excerpt=transcript_excerpt,
            caption=caption,
            ocr_text=ocr_text,
            diagram_labels=diagram_labels,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_path=frame_path,
        )
        path = Path(frame_path) if frame_path else None
        if path is None or not path.is_file():
            return base
        try:
            cv_colors = _opencv_dominant_colors(path)
            edge_density = _opencv_edge_density(path)
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("opencv analytics error: %s", exc)
            return base
        merged_colors = tuple(dict.fromkeys(base.dominant_colors + cv_colors))
        is_diagram = base.is_diagram or edge_density >= 0.08
        visual_features = _build_features(
            motion_level=base.motion_level,
            dominant_colors=merged_colors,
            is_diagram=is_diagram,
            tags=tags,
        )
        if edge_density >= 0.08 and "edge_dense" not in visual_features:
            visual_features = visual_features + ("edge_dense",)
        summary = base.analytics_summary
        if cv_colors:
            summary = f"{summary} | cv colors: {', '.join(cv_colors[:3])}"
        if edge_density >= 0.08:
            summary = f"{summary} | edge density {edge_density:.2f}"
        return base.model_copy(
            update={
                "dominant_colors": merged_colors,
                "is_diagram": is_diagram,
                "visual_features": visual_features,
                "analytics_summary": summary,
            }
        )


def get_vision_analyzer(backend: str = "auto") -> VisionAnalyzer:
    """Resolve vision analyzer: stub, opencv, or auto (opencv when cv2 exists)."""
    normalized = backend.lower().strip()
    if normalized in {"stub", "metadata", "offline"}:
        return StubVisionAnalyzer()
    if normalized in {"opencv", "auto", "hybrid"}:
        if normalized == "auto" and not _opencv_available():
            return StubVisionAnalyzer()
        return OpenCvVisionAnalyzer(StubVisionAnalyzer())
    raise ValueError(f"unsupported vision analytics backend: {backend}")


def enrich_visual_cue(
    visual: VisualCue,
    *,
    analyzer: VisionAnalyzer,
    transcript_excerpt: str = "",
) -> VisualCue:
    """Return visual cue with motion, color, and diagram analytics."""
    analytics = analyzer.analyze(
        description=visual.description,
        tags=visual.tags,
        transcript_excerpt=transcript_excerpt,
        caption=visual.caption,
        ocr_text=visual.ocr_text,
        diagram_labels=visual.diagram_labels,
        frame_start=visual.frame_start,
        frame_end=visual.frame_end,
        frame_path=visual.frame_path,
    )
    merged_tags = tuple(dict.fromkeys(visual.tags + analytics.visual_features))
    return visual.model_copy(
        update={
            "motion_level": analytics.motion_level,
            "dominant_colors": analytics.dominant_colors,
            "is_diagram": analytics.is_diagram,
            "visual_features": analytics.visual_features,
            "analytics_summary": analytics.analytics_summary,
            "tags": merged_tags,
        }
    )


def enrich_synced_evidence_analytics(
    row: SyncedEvidence,
    *,
    analyzer: VisionAnalyzer,
) -> SyncedEvidence:
    """Attach visual analytics to one synced evidence row."""
    if row.visual is None:
        return row
    visual = enrich_visual_cue(row.visual, analyzer=analyzer, transcript_excerpt=row.transcript_excerpt)
    bonus = 0.0
    if visual.analytics_summary:
        bonus += 0.03
    if visual.visual_features:
        bonus += 0.03
    if visual.dominant_colors:
        bonus += 0.02
    if visual.is_diagram:
        bonus += 0.02
    confidence = min(1.0, round(row.confidence + bonus, 4))
    return row.model_copy(update={"visual": visual, "confidence": confidence})
