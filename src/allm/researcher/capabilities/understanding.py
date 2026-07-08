"""L2 — Understanding capabilities (KDP + sample packaging)."""

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
from allm.researcher.packages import (
    package_from_book_dir,
    package_from_repository,
    package_from_samples_jsonl,
    package_from_workshop_dir,
)

logger = get_logger("researcher.understanding")


class PackageUnderstandingCapability:
    """L2 — turn discoveries into Knowledge Packages."""

    level = 2
    name = "understanding.package"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        built = 0

        for discovery in pipeline.discoveries:
            if discovery.kind == "workshop" and discovery.paths:
                package = package_from_workshop_dir(
                    cfg.workshop_dir,
                    provider=discovery.provider_id,
                    max_files=cfg.workshop_max_files,
                    curriculum_topic=cfg.workshop_curriculum_topic,
                )
                pipeline.packages.append(package)
                pipeline.conflicts_detected += len(package.conflicts)
                built += 1
                logger.info(
                    "understanding.package: workshop %s concepts=%d conflicts=%d",
                    package.id,
                    len(package.concepts),
                    len(package.conflicts),
                )
            elif discovery.kind == "software" and discovery.paths:
                package = package_from_samples_jsonl(
                    Path(discovery.paths[0]),
                    provider=discovery.provider_id,
                    title=discovery.title,
                )
                pipeline.packages.append(package)
                built += 1
                logger.info(
                    "understanding.package: software %s concepts=%d",
                    package.id,
                    len(package.concepts),
                )
            elif discovery.kind == "repository" and discovery.paths:
                package = package_from_repository(
                    cfg.repository_dir,
                    provider=discovery.provider_id,
                    max_files=cfg.repository_max_files,
                )
                pipeline.packages.append(package)
                pipeline.conflicts_detected += len(package.conflicts)
                built += 1
                logger.info(
                    "understanding.package: repository %s concepts=%d conflicts=%d",
                    package.id,
                    len(package.concepts),
                    len(package.conflicts),
                )
            elif discovery.kind == "book" and discovery.paths:
                package = package_from_book_dir(
                    cfg.book_dir,
                    provider=discovery.provider_id,
                    max_files=cfg.book_max_files,
                    max_pages=cfg.book_max_pages,
                    curriculum_topic=cfg.book_curriculum_topic,
                    pdf_backend=cfg.book_pdf_backend,
                )
                pipeline.packages.append(package)
                pipeline.conflicts_detected += len(package.conflicts)
                built += 1
                logger.info(
                    "understanding.package: book %s concepts=%d conflicts=%d",
                    package.id,
                    len(package.concepts),
                    len(package.conflicts),
                )

        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=built,
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"package_count": built},
        )
