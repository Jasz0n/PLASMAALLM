"""L3 — Cross-source verification: workshop transcripts vs Keshe books."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.cross_source import align_workshop_and_book, find_packages_by_provider

logger = get_logger("researcher.cross_source")


class CrossSourceVerificationCapability:
    """Align workshop and book concepts; surface agreement for Teacher/KEL."""

    level = 3
    name = "verification.cross_source"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_cross_source_verification:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="cross-source verification disabled",
                ),
            )

        target = pipeline.verified_packages or pipeline.packages
        workshop, book = find_packages_by_provider(target)
        if workshop is None or book is None:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="need workshop and book packages",
                ),
            )

        report = align_workshop_and_book(
            workshop,
            book,
            min_overlap=cfg.cross_source_min_overlap,
        )
        pipeline.cross_source_report = report
        if report.aligned_count:
            pipeline.proposal_hints.append(report.summary)

        verified: list = []
        for package in pipeline.verified_packages or pipeline.packages:
            if package.provider != "keshe-books" or report.aligned_count == 0:
                verified.append(package)
                continue
            boost = min(0.08, 0.02 * report.aligned_count)
            verified.append(
                package.model_copy(
                    update={"confidence": round(min(1.0, package.confidence + boost), 4)}
                )
            )
        if pipeline.verified_packages:
            pipeline.verified_packages = verified

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "verification.cross_source: aligned=%d workshop_only=%d book_only=%d",
            report.aligned_count,
            len(report.workshop_only),
            len(report.book_only),
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=report.aligned_count,
                duration_ms=round(elapsed, 2),
                notes=report.summary[:120],
            ),
            artifacts={"cross_source": report},
        )
