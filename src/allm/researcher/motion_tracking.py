"""Temporal motion tracking across sampled frame sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.researcher.multimodal_types import SyncedEvidence, VisualCue
from allm.researcher.vision_analytics import _collect_text, _opencv_available

logger = get_logger("researcher.motion_tracking")

_ROTATION_HINTS = frozenset({"rotating", "rotation", "spin", "spinning", "magnet-rotation"})
_OSCILLATION_HINTS = frozenset({"chasing", "chase", "beat", "field-beat", "plasma-motion", "twist"})
_TRANSLATION_HINTS = frozenset({"repulsion", "interaction", "moving", "motion"})
_STATIC_HINTS = frozenset({"static", "still", "close-up", "close up"})


class MotionTrackResult(BaseModel):
    """Temporal motion observations for one visual cue span."""

    model_config = ConfigDict(frozen=True)

    motion_vector: str | None = None
    motion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    frame_sequence_paths: tuple[str, ...] = ()
    motion_summary: str = ""


@runtime_checkable
class MotionTracker(Protocol):
    """Track motion across a short frame sequence or fixture span."""

    def track(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        motion_level: str | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_sequence_paths: tuple[str, ...] = (),
    ) -> MotionTrackResult: ...


def _span_score(frame_start: int | None, frame_end: int | None) -> float:
    if frame_start is None or frame_end is None or frame_end <= frame_start:
        return 0.0
    return min(1.0, (frame_end - frame_start) / 60.0)


def _level_score(motion_level: str | None) -> float:
    if motion_level == "high":
        return 0.85
    if motion_level == "moderate":
        return 0.55
    if motion_level == "static":
        return 0.12
    return 0.0


def _infer_motion_vector(text: str, tags: tuple[str, ...]) -> str | None:
    normalized_tags = {tag.lower().replace("_", "-") for tag in tags}
    if normalized_tags & _STATIC_HINTS or any(token in text for token in _STATIC_HINTS):
        return "static"
    if normalized_tags & _ROTATION_HINTS or any(word in text for word in _ROTATION_HINTS):
        return "rotation"
    if normalized_tags & _OSCILLATION_HINTS or any(word in text for word in _OSCILLATION_HINTS):
        return "oscillation"
    if normalized_tags & _TRANSLATION_HINTS or any(word in text for word in _TRANSLATION_HINTS):
        return "translation"
    return None


def _merge_motion_score(
    *,
    span_score: float,
    level_score: float,
    measured_score: float | None = None,
) -> float:
    if measured_score is not None and measured_score > 0:
        return round(min(1.0, max(measured_score, span_score * 0.5, level_score * 0.5)), 4)
    base = max(span_score, level_score)
    if span_score > 0 and level_score > 0:
        base = min(1.0, (span_score + level_score) / 1.4)
    return round(base, 4)


def _opencv_sequence_motion(paths: tuple[str, ...]) -> tuple[float, str | None]:
    import cv2

    readable = [Path(path) for path in paths if Path(path).is_file()]
    if len(readable) < 2:
        return 0.0, None
    diffs: list[float] = []
    shifts: list[tuple[float, float]] = []
    for left, right in zip(readable, readable[1:]):
        img_left = cv2.imread(str(left), cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
        if img_left is None or img_right is None:
            continue
        small_left = cv2.resize(img_left, (64, 64))
        small_right = cv2.resize(img_right, (64, 64))
        diff = cv2.absdiff(small_left, small_right)
        diffs.append(float(diff.mean()) / 255.0)
        moments = cv2.moments(diff)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"] - 32.0
            cy = moments["m01"] / moments["m00"] - 32.0
            shifts.append((cx, cy))
    if not diffs:
        return 0.0, None
    score = min(1.0, sum(diffs) / len(diffs) * 2.5)
    if score < 0.08:
        return score, "static"
    if not shifts:
        return score, "oscillation"
    dx = sum(item[0] for item in shifts) / len(shifts)
    dy = sum(item[1] for item in shifts) / len(shifts)
    if abs(dx) > abs(dy) * 1.4:
        return score, "translation"
    if abs(dy) > abs(dx) * 1.4:
        return score, "rotation"
    return score, "oscillation"


class StubMotionTracker:
    """Offline tracker using fixture frame spans and metadata heuristics."""

    def track(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        motion_level: str | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_sequence_paths: tuple[str, ...] = (),
    ) -> MotionTrackResult:
        text = _collect_text(
            description=description,
            tags=tags,
            transcript_excerpt=transcript_excerpt,
            caption=None,
            ocr_text=None,
        )
        span_score = _span_score(frame_start, frame_end)
        level_score = _level_score(motion_level)
        motion_score = _merge_motion_score(span_score=span_score, level_score=level_score)
        motion_vector = _infer_motion_vector(text, tags)
        if motion_vector is None and motion_score >= 0.5:
            motion_vector = "oscillation"
        if motion_vector is None and motion_score > 0:
            motion_vector = "static"
        frame_span = ""
        if frame_start is not None and frame_end is not None:
            frame_span = f"frames {frame_start}-{frame_end}"
        parts = ["Span-tracked visual"]
        if frame_sequence_paths:
            parts[0] = f"Sequence tracked ({len(frame_sequence_paths)} frames)"
        if frame_span:
            parts.append(frame_span)
        if motion_vector:
            parts.append(f"vector: {motion_vector}")
        parts.append(f"score: {motion_score:.2f}")
        return MotionTrackResult(
            motion_vector=motion_vector,
            motion_score=motion_score,
            frame_sequence_paths=frame_sequence_paths,
            motion_summary=" — ".join(parts),
        )


class OpenCvMotionTracker:
    """Merge metadata heuristics with OpenCV frame-diff tracking when frames exist."""

    def __init__(self, fallback: MotionTracker | None = None) -> None:
        self._fallback = fallback or StubMotionTracker()

    def track(
        self,
        *,
        description: str,
        tags: tuple[str, ...] = (),
        transcript_excerpt: str = "",
        motion_level: str | None = None,
        frame_start: int | None = None,
        frame_end: int | None = None,
        frame_sequence_paths: tuple[str, ...] = (),
    ) -> MotionTrackResult:
        base = self._fallback.track(
            description=description,
            tags=tags,
            transcript_excerpt=transcript_excerpt,
            motion_level=motion_level,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_sequence_paths=frame_sequence_paths,
        )
        if len(frame_sequence_paths) < 2:
            return base
        try:
            measured_score, measured_vector = _opencv_sequence_motion(frame_sequence_paths)
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("opencv motion tracking error: %s", exc)
            return base
        span_score = _span_score(frame_start, frame_end)
        level_score = _level_score(motion_level)
        motion_score = _merge_motion_score(
            span_score=span_score,
            level_score=level_score,
            measured_score=measured_score,
        )
        motion_vector = measured_vector or base.motion_vector
        summary = base.motion_summary
        if measured_score > 0:
            summary = f"{summary} | cv motion {measured_score:.2f}"
        if measured_vector:
            summary = f"{summary} | cv vector {measured_vector}"
        return base.model_copy(
            update={
                "motion_vector": motion_vector,
                "motion_score": motion_score,
                "motion_summary": summary,
            }
        )


def get_motion_tracker(backend: str = "auto") -> MotionTracker:
    """Resolve motion tracker: stub, opencv, or auto (opencv when cv2 exists)."""
    normalized = backend.lower().strip()
    if normalized in {"stub", "metadata", "offline"}:
        return StubMotionTracker()
    if normalized in {"opencv", "auto", "hybrid"}:
        if normalized == "auto" and not _opencv_available():
            return StubMotionTracker()
        return OpenCvMotionTracker(StubMotionTracker())
    raise ValueError(f"unsupported motion tracking backend: {backend}")


def enrich_visual_motion(
    visual: VisualCue,
    *,
    tracker: MotionTracker,
    transcript_excerpt: str = "",
    frame_sequence_paths: tuple[str, ...] = (),
) -> VisualCue:
    """Return visual cue with temporal motion tracking fields."""
    paths = frame_sequence_paths or visual.frame_sequence_paths
    result = tracker.track(
        description=visual.description,
        tags=visual.tags,
        transcript_excerpt=transcript_excerpt,
        motion_level=visual.motion_level,
        frame_start=visual.frame_start,
        frame_end=visual.frame_end,
        frame_sequence_paths=paths,
    )
    merged_features = tuple(
        dict.fromkeys(
            visual.visual_features
            + (
                (f"motion_vector_{result.motion_vector}",)
                if result.motion_vector
                else ()
            )
            + ((f"motion_score_{int(result.motion_score * 100)}",) if result.motion_score > 0 else ())
        )
    )
    return visual.model_copy(
        update={
            "motion_vector": result.motion_vector,
            "motion_score": result.motion_score,
            "frame_sequence_paths": result.frame_sequence_paths,
            "motion_summary": result.motion_summary,
            "visual_features": merged_features,
        }
    )


def enrich_synced_evidence_motion(
    row: SyncedEvidence,
    *,
    tracker: MotionTracker,
    frame_sequence_paths: tuple[str, ...] = (),
) -> SyncedEvidence:
    """Attach temporal motion tracking to one synced evidence row."""
    if row.visual is None:
        return row
    visual = enrich_visual_motion(
        row.visual,
        tracker=tracker,
        transcript_excerpt=row.transcript_excerpt,
        frame_sequence_paths=frame_sequence_paths,
    )
    bonus = 0.0
    if visual.motion_summary:
        bonus += 0.03
    if visual.motion_vector and visual.motion_vector != "static":
        bonus += 0.03
    if visual.motion_score and visual.motion_score >= 0.5:
        bonus += 0.02
    confidence = min(1.0, round(row.confidence + bonus, 4))
    return row.model_copy(update={"visual": visual, "confidence": confidence})
