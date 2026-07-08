"""L2 — Visual distillation for Teacher handoff."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.visual_distillation import (
    attach_distilled_visuals,
    briefs_for_provider,
    distill_visual_evidence,
)

logger = get_logger("researcher.visual_distill")


class VisualDistillationCapability:
    """Distill enriched multimodal evidence into Teacher-ready visual briefs."""

    level = 2
    name = "understanding.visual.distill"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_visual_distillation:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="visual distillation disabled",
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

        rows = list(pipeline.multimodal_synced)
        book_rows = [row for row in rows if row.source_id.startswith("book:")]
        workshop_rows = [row for row in rows if not row.source_id.startswith("book:")]

        book_briefs = distill_visual_evidence(
            book_rows,
            max_images=cfg.visual_distillation_max_images,
            max_questions=cfg.visual_distillation_max_questions,
        )
        workshop_briefs = distill_visual_evidence(
            workshop_rows,
            max_images=cfg.visual_distillation_max_images,
            max_questions=cfg.visual_distillation_max_questions,
        )
        briefs = book_briefs + workshop_briefs
        if not briefs:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no visual briefs produced",
                ),
            )

        allowed_topics = {cfg.workshop_curriculum_topic, cfg.book_curriculum_topic}
        updated = 0
        for index, package in enumerate(pipeline.packages):
            topic = package.curriculum_topic or cfg.workshop_curriculum_topic
            if topic and allowed_topics and topic not in allowed_topics:
                continue
            provider_briefs = briefs_for_provider(briefs, package.provider)
            if not provider_briefs:
                continue
            pipeline.packages[index] = attach_distilled_visuals(package, provider_briefs)
            updated += 1

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "understanding.visual.distill: briefs=%d packages=%d",
            len(briefs),
            updated,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(briefs),
                duration_ms=round(elapsed, 2),
                notes=f"packages={updated}",
            ),
            artifacts={"briefs": briefs},
        )
