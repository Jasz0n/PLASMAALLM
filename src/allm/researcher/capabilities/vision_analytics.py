"""L2 — Vision analytics: motion, color, and diagram detection on synced evidence."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.multimodal import attach_multimodal_evidence
from allm.researcher.vision_analytics import enrich_synced_evidence_analytics, get_vision_analyzer

logger = get_logger("researcher.vision_analytics")


class VisionAnalyticsCapability:
    """Detect motion, color, and diagram structure on enriched visual cues."""

    level = 2
    name = "understanding.vision.analytics"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_vision_analytics:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="vision analytics disabled",
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

        analyzer = get_vision_analyzer(cfg.vision_analytics_backend)
        enriched_rows = [
            enrich_synced_evidence_analytics(row, analyzer=analyzer)
            for row in pipeline.multimodal_synced
        ]
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
        analyzed = sum(
            1
            for row in enriched_rows
            if row.visual and (row.visual.visual_features or row.visual.analytics_summary)
        )
        diagrams = sum(1 for row in enriched_rows if row.visual and row.visual.is_diagram)
        logger.info(
            "understanding.vision.analytics: analyzed=%d diagrams=%d",
            analyzed,
            diagrams,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=analyzed,
                duration_ms=round(elapsed, 2),
                notes=f"diagrams={diagrams}",
            ),
            artifacts={"enriched": enriched_rows},
        )
