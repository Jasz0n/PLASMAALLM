"""L2 — Vision enrichment: frame extraction + captions on synced evidence."""

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
from allm.researcher.frame_extractor import extract_frame_for_source
from allm.researcher.multimodal import attach_multimodal_evidence
from allm.researcher.vision_caption import enrich_synced_evidence, get_vision_captioner

logger = get_logger("researcher.vision")


class VisionEnrichmentCapability:
    """Extract frames (ffmpeg) and caption visual cues on synced evidence."""

    level = 2
    name = "understanding.vision"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_vision_captions:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="vision captions disabled",
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

        cache_dir = cfg.frames_cache_dir
        if cache_dir is None and cfg.workshop_dir is not None:
            cache_dir = Path(cfg.workshop_dir) / ".frames_cache"
        if cache_dir is None:
            cache_dir = Path(".frames_cache")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        captioner = get_vision_captioner(
            cfg.vision_caption_backend,
            ollama_model=cfg.vision_ollama_model,
        )
        enriched_rows = []
        frames_extracted = 0

        for row in pipeline.multimodal_synced:
            frame_path = None
            if row.visual is not None and row.visual.frame_path:
                frame_path = row.visual.frame_path
            elif cfg.video_dir is not None:
                extracted = extract_frame_for_source(
                    source_id=row.source_id,
                    transcript_name=f"{row.source_id}.txt",
                    timestamp_sec=row.timestamp_sec,
                    video_dir=cfg.video_dir,
                    cache_dir=cache_dir,
                )
                if extracted is not None:
                    frame_path = str(extracted)
                    frames_extracted += 1
            enriched_rows.append(
                enrich_synced_evidence(row, captioner=captioner, frame_path=frame_path)
            )

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
        captioned = sum(1 for row in enriched_rows if row.visual and row.visual.caption)
        logger.info(
            "understanding.vision: captioned=%d frames=%d",
            captioned,
            frames_extracted,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=captioned,
                duration_ms=round(elapsed, 2),
                notes=f"frames={frames_extracted}",
            ),
            artifacts={"enriched": enriched_rows},
        )
