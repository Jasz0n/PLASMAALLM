"""L2 — Object identity persistence across workshop sources."""

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
from allm.researcher.object_identity import enrich_object_identities

logger = get_logger("researcher.object_identity")


class ObjectIdentityCapability:
    """Persist object identities across multiple workshop video sources."""

    level = 2
    name = "understanding.vision.identity"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_object_identity:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="object identity disabled",
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
        if cfg.enable_motion_continuity and not any(row.motion_track_id for row in rows):
            rows, _tracks = enrich_synced_evidence_continuity(
                rows,
                min_score=cfg.motion_continuity_min_score,
            )

        enriched_rows, identities = enrich_object_identities(
            rows,
            min_score=cfg.object_identity_min_score,
            curriculum_topic=cfg.workshop_curriculum_topic,
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
        cross_source = sum(1 for row in enriched_rows if row.linked_source_ids)
        multi_source = sum(1 for record in identities if len(record.source_ids) >= 2)
        logger.info(
            "understanding.vision.identity: cross_source=%d identities=%d multi=%d",
            cross_source,
            len(identities),
            multi_source,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=cross_source,
                duration_ms=round(elapsed, 2),
                notes=f"identities={len(identities)} multi={multi_source}",
            ),
            artifacts={"enriched": enriched_rows, "identities": identities},
        )
