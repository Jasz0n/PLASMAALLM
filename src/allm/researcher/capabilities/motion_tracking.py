"""L2 — Temporal motion tracking across sampled frame sequences."""

from __future__ import annotations

import time
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.frame_extractor import extract_frame_sequence_for_source
from allm.researcher.motion_tracking import enrich_synced_evidence_motion, get_motion_tracker
from allm.researcher.multimodal import attach_multimodal_evidence

logger = get_logger("researcher.motion")


class MotionTrackingCapability:
    """Track motion across frame sequences for synced visual evidence."""

    level = 2
    name = "understanding.vision.motion"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_motion_tracking:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="motion tracking disabled",
                ),
            )

        if not pipeline.multimodal_synced:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no synced evidence",
                ),
            )

        cache_dir = cfg.frames_cache_dir
        if cache_dir is None and cfg.workshop_dir is not None:
            cache_dir = Path(cfg.workshop_dir) / ".frames_cache"
        if cache_dir is None:
            cache_dir = Path(".frames_cache")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        tracker = get_motion_tracker(cfg.motion_tracking_backend)
        enriched_rows = []
        sequences_extracted = 0

        for row in pipeline.multimodal_synced:
            frame_paths: tuple[str, ...] = ()
            if row.visual is not None and cfg.video_dir is not None:
                extracted = extract_frame_sequence_for_source(
                    source_id=row.source_id,
                    transcript_name=f"{row.source_id}.txt",
                    timestamp_sec=row.timestamp_sec,
                    frame_start=row.visual.frame_start,
                    frame_end=row.visual.frame_end,
                    video_dir=cfg.video_dir,
                    cache_dir=cache_dir,
                    sample_count=cfg.motion_tracking_samples,
                    fps=cfg.motion_tracking_fps,
                )
                if extracted:
                    frame_paths = tuple(str(path) for path in extracted)
                    sequences_extracted += 1
            enriched_rows.append(
                enrich_synced_evidence_motion(
                    row,
                    tracker=tracker,
                    frame_sequence_paths=frame_paths,
                )
            )

        pipeline.multimodal_synced = enriched_rows

        for index, package in enumerate(pipeline.packages):
            if not package.multimodal_evidence:
                continue
            pipeline.packages[index] = attach_multimodal_evidence(
                package,
                list(enriched_rows),
                curriculum_topic=cfg.workshop_curriculum_topic,
            )

        elapsed = (time.perf_counter() - started) * 1000
        tracked = sum(
            1 for row in enriched_rows if row.visual and row.visual.motion_summary
        )
        moving = sum(
            1
            for row in enriched_rows
            if row.visual and row.visual.motion_vector not in {None, "static"}
        )
        logger.info(
            "understanding.vision.motion: tracked=%d moving=%d sequences=%d",
            tracked,
            moving,
            sequences_extracted,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=tracked,
                duration_ms=round(elapsed, 2),
                notes=f"moving={moving} sequences={sequences_extracted}",
            ),
            artifacts={"enriched": enriched_rows},
        )
