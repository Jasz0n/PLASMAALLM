"""L3 — Teacher-approved student visual export."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.student_visual_export import attach_student_visual_packages
from allm.teacher.visual_export import (
    approve_visual_brief,
    auto_approve_briefs,
    export_approved_briefs,
    resolve_visual_approvals,
)

logger = get_logger("researcher.visual_export")


class VisualExportCapability:
    """Export Teacher-approved visual subsets for student delivery."""

    level = 3
    name = "understanding.visual.export"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_visual_export:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="visual export disabled",
                ),
            )

        if not (pipeline.verified_packages or pipeline.packages):
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no packages",
                ),
            )

        exported_total = 0
        updated = 0
        all_exports: list = []
        target = pipeline.verified_packages if pipeline.verified_packages else pipeline.packages

        for index, package in enumerate(list(target)):
            briefs = package.distilled_visual_briefs
            if not briefs:
                continue

            if cfg.visual_export_auto_approve:
                approvals = auto_approve_briefs(
                    briefs,
                    min_confidence=cfg.visual_export_min_confidence,
                    max_images=cfg.visual_export_max_images,
                    max_questions=cfg.visual_export_max_questions,
                )
                if cfg.visual_export_persist_approvals:
                    from allm.teacher.visual_approval_store import VisualApprovalStore

                    approval_store = VisualApprovalStore(ctx.store)
                    for approval in approvals:
                        approval_store.save(approval, reason="auto-approve")
            else:
                approvals = resolve_visual_approvals(
                    briefs,
                    store=ctx.store,
                    auto_approve=False,
                    max_images=cfg.visual_export_max_images,
                    max_questions=cfg.visual_export_max_questions,
                    persist=cfg.visual_export_persist_approvals,
                )

            if package.provider == "keshe-books":
                topic = package.curriculum_topic or cfg.book_curriculum_topic
            else:
                topic = package.curriculum_topic or cfg.workshop_curriculum_topic
            exports = export_approved_briefs(
                briefs,
                approvals,
                curriculum_topic=topic or "",
            )
            if not exports:
                continue

            updated_package = attach_student_visual_packages(package, exports)
            if pipeline.verified_packages:
                pipeline.verified_packages[index] = updated_package
            pipeline.packages[index] = updated_package
            exported_total += len(exports)
            updated += 1
            all_exports.extend(exports)

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "understanding.visual.export: exports=%d packages=%d",
            exported_total,
            updated,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=exported_total,
                duration_ms=round(elapsed, 2),
                notes=f"packages={updated}",
            ),
            artifacts={"exports": all_exports},
        )
