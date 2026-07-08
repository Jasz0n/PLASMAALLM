"""L2 — OCR enrichment: read diagram text from extracted frames."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.frame_ocr import enrich_synced_evidence_ocr, get_frame_ocr
from allm.researcher.multimodal import attach_multimodal_evidence

logger = get_logger("researcher.ocr")


class OcrEnrichmentCapability:
    """Run OCR on visual cues that already have extracted frame paths."""

    level = 2
    name = "understanding.ocr"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_frame_ocr:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="frame OCR disabled",
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

        reader = get_frame_ocr(
            cfg.ocr_backend,
            ollama_model=cfg.ocr_ollama_model,
        )
        enriched_rows = []
        frames_read = 0

        for row in pipeline.multimodal_synced:
            if row.visual is not None and row.visual.frame_path:
                frames_read += 1
            enriched_rows.append(enrich_synced_evidence_ocr(row, ocr=reader))

        pipeline.multimodal_synced = enriched_rows

        for index, package in enumerate(pipeline.packages):
            if not package.multimodal_evidence:
                continue
            topic = package.curriculum_topic or cfg.workshop_curriculum_topic
            pipeline.packages[index] = attach_multimodal_evidence(
                package,
                list(enriched_rows),
                curriculum_topic=topic,
            )

        elapsed = (time.perf_counter() - started) * 1000
        ocr_hits = sum(
            1 for row in enriched_rows if row.visual and (row.visual.ocr_text or row.visual.diagram_labels)
        )
        logger.info(
            "understanding.ocr: ocr_hits=%d frames=%d",
            ocr_hits,
            frames_read,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=ocr_hits,
                duration_ms=round(elapsed, 2),
                notes=f"frames={frames_read}",
            ),
            artifacts={"enriched": enriched_rows},
        )
