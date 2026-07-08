"""L1/L2 — Multimodal discovery and transcript synchronization."""

from __future__ import annotations

import time
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    DiscoveryArtifact,
    PipelineState,
)
from allm.researcher.multimodal import (
    attach_multimodal_evidence,
    discover_video_fixtures,
    sync_fixtures_with_workshop_dir,
)
from allm.researcher.video_decoder import ensure_workshop_fixtures

logger = get_logger("researcher.multimodal")


class VideoDiscoveryCapability:
    """L1 — discover video timeline fixtures (offline stand-in for video decoder)."""

    level = 1
    name = "discovery.video"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        fixture_dir = cfg.video_fixture_dir
        if fixture_dir is None and cfg.workshop_dir is not None and cfg.auto_generate_video_fixtures:
            fixture_dir = Path(cfg.workshop_dir) / ".visual_cache"
        if fixture_dir is None or (not fixture_dir.is_dir() and not cfg.auto_generate_video_fixtures):
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no video fixture dir",
                ),
            )

        if cfg.auto_generate_video_fixtures and cfg.workshop_dir is not None:
            ensure_workshop_fixtures(
                cfg.workshop_dir,
                fixture_dir,
                video_dir=cfg.video_dir,
                curriculum_topic=cfg.workshop_curriculum_topic,
            )
            fixture_dir.mkdir(parents=True, exist_ok=True)

        if not fixture_dir.is_dir():
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="fixture dir missing",
                ),
            )

        fixtures = discover_video_fixtures(fixture_dir)
        if not fixtures:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no fixtures found",
                ),
            )

        paths = tuple(str(path) for path in sorted(fixture_dir.glob("*.json")))
        artifact = DiscoveryArtifact(
            provider_id="kids-video-fixtures",
            kind="video",
            paths=paths,
            reputation_score=0.75,
            title=f"Workshop video timelines ({len(fixtures)} fixtures)",
        )
        pipeline.discoveries.append(artifact)
        pipeline.providers_evaluated += 1
        pipeline.video_fixtures = fixtures

        elapsed = (time.perf_counter() - started) * 1000
        logger.info("discovery.video: fixtures=%d", len(fixtures))
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(fixtures),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"fixtures": fixtures},
        )


class MultimodalSyncCapability:
    """L2 — synchronize video cues with workshop transcripts and attach to packages."""

    level = 2
    name = "understanding.sync"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        fixtures = pipeline.video_fixtures
        if not fixtures:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no video fixtures in pipeline",
                ),
            )

        workshop_dir = ctx.config.workshop_dir
        if workshop_dir is None or not workshop_dir.is_dir():
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no workshop dir for sync",
                ),
            )

        synced = sync_fixtures_with_workshop_dir(fixtures, workshop_dir)
        pipeline.multimodal_synced = synced

        updated: list = []
        for index, package in enumerate(pipeline.packages):
            attached = attach_multimodal_evidence(
                package,
                synced,
                curriculum_topic=ctx.config.workshop_curriculum_topic,
            )
            if attached.multimodal_evidence:
                pipeline.packages[index] = attached
                updated.append(attached)

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "understanding.sync: synced=%d packages_updated=%d",
            len(synced),
            len(updated),
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(synced),
                duration_ms=round(elapsed, 2),
                notes=f"{len(updated)} packages",
            ),
            artifacts={"synced": synced},
        )
