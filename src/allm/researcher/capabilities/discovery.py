"""L1 — Discovery capabilities."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    DiscoveryArtifact,
    PipelineState,
)
from allm.researcher.providers import (
    BookProvider,
    RepositoryProvider,
    SoftwareFixtureProvider,
    WorkshopProvider,
)

logger = get_logger("researcher.discovery")


class WorkshopDiscoveryCapability:
    """L1 — discover kids workshop transcript sources."""

    level = 1
    name = "discovery.workshop"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if cfg.workshop_dir is None or not cfg.workshop_dir.is_dir():
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    success=True,
                    yield_count=0,
                    duration_ms=0.0,
                    notes="no workshop dir",
                ),
            )

        hints = ctx.strategy_hints
        if hints is not None and "kids-workshops" in getattr(hints, "skip_providers", ()):
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="skipped by strategy hints",
                ),
            )

        provider = WorkshopProvider(cfg.workshop_dir)
        rep = provider.reputation()
        paths = provider.discover()
        if cfg.workshop_max_files is not None:
            paths = paths[: cfg.workshop_max_files]
        artifact = DiscoveryArtifact(
            provider_id=provider.provider_id,
            kind=provider.kind,
            paths=tuple(str(p) for p in paths),
            reputation_score=rep.score,
            title=f"Kids plasma workshops ({len(paths)} files)",
        )
        pipeline.discoveries.append(artifact)
        pipeline.providers_evaluated += 1
        logger.info(
            "discovery.workshop: %s reputation=%.2f files=%d",
            provider.provider_id,
            rep.score,
            len(paths),
        )
        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(paths),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"discovery": artifact},
        )


class SoftwareDiscoveryCapability:
    """L1 — discover software fixture sources."""

    level = 1
    name = "discovery.software"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if cfg.software_samples is None or not cfg.software_samples.is_file():
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no software samples",
                ),
            )

        hints = ctx.strategy_hints
        if hints is not None and "software-fixture" in getattr(hints, "skip_providers", ()):
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="skipped by strategy hints",
                ),
            )

        provider = SoftwareFixtureProvider(cfg.software_samples)
        rep = provider.reputation()
        paths = provider.discover()
        artifact = DiscoveryArtifact(
            provider_id=provider.provider_id,
            kind=provider.kind,
            paths=tuple(str(p) for p in paths),
            reputation_score=rep.score,
            title="AI-friendly software development",
        )
        pipeline.discoveries.append(artifact)
        pipeline.providers_evaluated += 1
        logger.info(
            "discovery.software: %s reputation=%.2f",
            provider.provider_id,
            rep.score,
        )
        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(paths),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"discovery": artifact},
        )


class BookDiscoveryCapability:
    """L1 — discover Keshe foundation book PDFs."""

    level = 1
    name = "discovery.book"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if cfg.book_dir is None or not cfg.book_dir.is_dir():
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    success=True,
                    yield_count=0,
                    duration_ms=0.0,
                    notes="no book dir",
                ),
            )

        hints = ctx.strategy_hints
        if hints is not None and "keshe-books" in getattr(hints, "skip_providers", ()):
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="skipped by strategy hints",
                ),
            )

        provider = BookProvider(cfg.book_dir)
        rep = provider.reputation()
        paths = provider.discover()
        if cfg.book_max_files is not None:
            paths = paths[: cfg.book_max_files]
        artifact = DiscoveryArtifact(
            provider_id=provider.provider_id,
            kind=provider.kind,
            paths=tuple(str(p) for p in paths),
            reputation_score=rep.score,
            title=f"Keshe foundation books ({len(paths)} pdf)",
        )
        pipeline.discoveries.append(artifact)
        pipeline.providers_evaluated += 1
        logger.info(
            "discovery.book: %s reputation=%.2f files=%d",
            provider.provider_id,
            rep.score,
            len(paths),
        )
        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(paths),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"discovery": artifact},
        )


class RepositoryDiscoveryCapability:
    """L1 — discover a real software repository (Roadmap M49)."""

    level = 1
    name = "discovery.repository"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if cfg.repository_dir is None or not cfg.repository_dir.is_dir():
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no repository configured",
                ),
            )

        provider = RepositoryProvider(
            cfg.repository_dir, max_files=cfg.repository_max_files
        )
        rep = provider.reputation()
        paths = provider.discover()
        artifact = DiscoveryArtifact(
            provider_id=provider.provider_id,
            kind=provider.kind,
            paths=tuple(str(p) for p in paths),
            reputation_score=rep.score,
            title=f"Repository {cfg.repository_dir.name} ({len(paths)} files)",
        )
        pipeline.discoveries.append(artifact)
        pipeline.providers_evaluated += 1
        logger.info(
            "discovery.repository: %s reputation=%.2f files=%d",
            provider.provider_id,
            rep.score,
            len(paths),
        )
        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(paths),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"discovery": artifact},
        )
