"""L2 — Audio enrichment: clip extraction + feature analysis on synced evidence."""

from __future__ import annotations

import time
from pathlib import Path

from allm.core.logging import get_logger
from allm.researcher.audio_analysis import enrich_synced_evidence_audio, get_audio_analyzer
from allm.researcher.audio_extractor import extract_audio_for_source
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.multimodal import attach_multimodal_evidence

logger = get_logger("researcher.audio")


class AudioEnrichmentCapability:
    """Extract audio clips (ffmpeg) and analyze cues on synced evidence."""

    level = 2
    name = "understanding.audio"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_audio_analysis:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="audio analysis disabled",
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

        cache_dir = cfg.audio_cache_dir
        if cache_dir is None and cfg.workshop_dir is not None:
            cache_dir = Path(cfg.workshop_dir) / ".audio_cache"
        if cache_dir is None:
            cache_dir = Path(".audio_cache")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        analyzer = get_audio_analyzer(cfg.audio_analysis_backend)
        enriched_rows = []
        clips_extracted = 0

        for row in pipeline.multimodal_synced:
            clip_path = None
            if cfg.video_dir is not None and row.audio is not None:
                extracted = extract_audio_for_source(
                    source_id=row.source_id,
                    transcript_name=f"{row.source_id}.txt",
                    timestamp_sec=row.timestamp_sec,
                    video_dir=cfg.video_dir,
                    cache_dir=cache_dir,
                    duration_sec=cfg.audio_clip_duration_sec,
                )
                if extracted is not None:
                    clip_path = str(extracted)
                    clips_extracted += 1
            enriched_rows.append(
                enrich_synced_evidence_audio(row, analyzer=analyzer, clip_path=clip_path)
            )

        pipeline.multimodal_synced = enriched_rows

        for index, package in enumerate(pipeline.packages):
            if not package.multimodal_evidence:
                continue
            pipeline.packages[index] = attach_multimodal_evidence(
                package,
                list(enriched_rows),
                curriculum_topic=cfg.workshop_curriculum_topic,
            )

        elapsed = (time.perf_counter() - started) * 1000
        analyzed = sum(1 for row in enriched_rows if row.audio and row.audio.features)
        logger.info(
            "understanding.audio: analyzed=%d clips=%d",
            analyzed,
            clips_extracted,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=analyzed,
                duration_ms=round(elapsed, 2),
                notes=f"clips={clips_extracted}",
            ),
            artifacts={"enriched": enriched_rows},
        )
