"""L2 — Extract and sync book PDF figures into multimodal evidence."""

from __future__ import annotations

import time
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.book_evidence import build_book_synced_evidence
from allm.researcher.book_images import book_images_cache_dir, extract_pdf_images
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.multimodal import attach_multimodal_evidence

logger = get_logger("researcher.book_images")


class BookImagesCapability:
    """Extract PDF figures and attach them to book Knowledge Packages."""

    level = 2
    name = "understanding.book.images"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_book_images:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="book images disabled",
                ),
            )

        if cfg.book_dir is None or not cfg.book_dir.is_dir():
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no book dir",
                ),
            )

        book_paths: list[Path] = []
        for discovery in pipeline.discoveries:
            if discovery.kind == "book" and discovery.paths:
                book_paths.extend(Path(path) for path in discovery.paths)
        if not book_paths:
            book_paths = sorted(cfg.book_dir.glob("*.pdf"))
            if cfg.book_max_files is not None:
                book_paths = book_paths[: cfg.book_max_files]

        if not book_paths:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no book pdfs",
                ),
            )

        cache_dir = book_images_cache_dir(cfg.book_dir, override=cfg.book_images_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        artifacts = []
        for path in book_paths:
            artifacts.extend(
                extract_pdf_images(
                    path,
                    cache_dir,
                    max_pages=cfg.book_max_pages,
                    max_images=cfg.book_max_images,
                    pdf_backend=cfg.book_pdf_backend,
                )
            )

        if not artifacts:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no figures extracted",
                ),
            )

        evidence = list(build_book_synced_evidence(artifacts))
        pipeline.multimodal_synced.extend(evidence)

        updated = 0
        for index, package in enumerate(pipeline.packages):
            if package.provider != "keshe-books":
                continue
            pipeline.packages[index] = attach_multimodal_evidence(
                package,
                evidence,
                curriculum_topic=package.curriculum_topic or cfg.book_curriculum_topic,
            )
            updated += 1

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "understanding.book.images: figures=%d packages=%d",
            len(evidence),
            updated,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(evidence),
                duration_ms=round(elapsed, 2),
                notes=f"packages={updated}",
            ),
            artifacts={"figures": evidence},
        )
