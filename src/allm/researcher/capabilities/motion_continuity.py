"""L2 — Cross-cue motion continuity across a workshop timeline."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.motion_continuity import enrich_synced_evidence_continuity
from allm.researcher.multimodal import attach_multimodal_evidence

logger = get_logger("researcher.motion_continuity")


class MotionContinuityCapability:
    """Link visual cues that track the same object or motion thread over time."""

    level = 2
    name = "understanding.vision.continuity"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_motion_continuity:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="motion continuity disabled",
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

        enriched_rows, tracks = enrich_synced_evidence_continuity(
            list(pipeline.multimodal_synced),
            min_score=cfg.motion_continuity_min_score,
        )
        pipeline.multimodal_synced = enriched_rows

        for index, package in enumerate(pipeline.packages):
            if not package.multimodal_evidence:
                continue
            pipeline.packages[index] = attach_multimodal_evidence(
                package,
                enriched_rows,
                curriculum_topic=cfg.workshop_curriculum_topic,
            )

        elapsed = (time.perf_counter() - started) * 1000
        linked = sum(1 for row in enriched_rows if row.linked_cue_timestamps)
        multi_cue_tracks = sum(1 for track in tracks if len(track.timestamps) >= 2)
        logger.info(
            "understanding.vision.continuity: linked=%d tracks=%d multi=%d",
            linked,
            len(tracks),
            multi_cue_tracks,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=linked,
                duration_ms=round(elapsed, 2),
                notes=f"tracks={len(tracks)} multi={multi_cue_tracks}",
            ),
            artifacts={"enriched": enriched_rows, "tracks": tracks},
        )
