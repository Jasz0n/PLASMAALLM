"""LiveKit discovery, observation, and archival capabilities."""

from __future__ import annotations

import os
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
from allm.researcher.livekit_archive import archive_evidence_rows
from allm.researcher.livekit_observer import get_livekit_observer
from allm.researcher.livekit_tokens import credentials_for_stream, load_livekit_config
from allm.researcher.livekit_types import LiveKitCredentials, LiveStreamInfo
from allm.researcher.livekit_worker import get_livekit_worker
from allm.researcher.multimodal import attach_multimodal_evidence
from allm.researcher.social_stream_client import (
    fetch_active_streams,
    join_live_stream,
    leave_live_stream,
    load_livekit_fixture,
)

logger = get_logger("researcher.livekit")


def _discover_streams(ctx: CapabilityContext) -> list[LiveStreamInfo]:
    cfg = ctx.config
    streams: list[LiveStreamInfo] = []

    if cfg.social_api_base_url:
        streams.extend(fetch_active_streams(str(cfg.social_api_base_url)))

    if cfg.livekit_fixture_path is not None:
        streams.extend(load_livekit_fixture(cfg.livekit_fixture_path))

    if cfg.livekit_stream_ids:
        config = load_livekit_config()
        livekit_url = config.url if config is not None else os.environ.get("LIVEKIT_URL", "")
        for stream_id in cfg.livekit_stream_ids:
            if any(row.stream_id == stream_id for row in streams):
                continue
            streams.append(
                LiveStreamInfo(
                    stream_id=stream_id,
                    title=f"Live stream {stream_id}",
                    status="live",
                    livekit_room_name=stream_id,
                    livekit_url=livekit_url,
                    curriculum_topic=cfg.workshop_curriculum_topic,
                )
            )

    if cfg.livekit_topics:
        topic_set = {topic.lower() for topic in cfg.livekit_topics}
        streams = [
            stream
            for stream in streams
            if not stream.topic or stream.topic.lower() in topic_set
            or stream.curriculum_topic.lower() in topic_set
            or any(tag.lower() in topic_set for tag in stream.tags)
        ]

    return streams


def _resolve_credentials(
    stream: LiveStreamInfo,
    cfg,
) -> tuple[LiveKitCredentials | None, bool]:
    """Resolve join credentials; return (credentials, joined_via_social_api)."""
    identity = cfg.livekit_researcher_identity

    if cfg.social_api_base_url:
        joined = join_live_stream(
            str(cfg.social_api_base_url),
            stream.stream_id,
            identity,
            role="viewer",
        )
        if joined is not None:
            _, credentials = joined
            return credentials, True

    config = load_livekit_config()
    if config is not None:
        return credentials_for_stream(stream, config, identity=identity), False

    if stream.snapshots:
        return LiveKitCredentials(
            stream_id=stream.stream_id,
            url=stream.livekit_url,
            room_name=stream.livekit_room_name,
            token="",
            identity=identity,
        ), False

    return None, False


class LiveKitDiscoveryCapability:
    """L1 — discover live LiveKit workshop streams."""

    level = 1
    name = "discovery.livekit"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_livekit:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="livekit disabled",
                ),
            )

        streams = _discover_streams(ctx)
        live_streams = [stream for stream in streams if stream.status == "live"]
        pipeline.live_streams = live_streams

        if not live_streams:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no live streams",
                ),
            )

        artifact = DiscoveryArtifact(
            provider_id="livekit-streams",
            kind="livekit",
            paths=tuple(stream.stream_id for stream in live_streams),
            reputation_score=0.85,
            title=f"LiveKit streams ({len(live_streams)} live)",
        )
        pipeline.discoveries.append(artifact)
        pipeline.providers_evaluated += 1

        elapsed = (time.perf_counter() - started) * 1000
        logger.info("discovery.livekit: live_streams=%d", len(live_streams))
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(live_streams),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"streams": live_streams},
        )


class LiveStreamObserveCapability:
    """L2 — join LiveKit as observer and capture live multimodal evidence."""

    level = 2
    name = "understanding.livestream"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_livekit:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="livekit disabled",
                ),
            )

        streams = pipeline.live_streams
        if not streams:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no live streams in pipeline",
                ),
            )

        observer = get_livekit_observer(cfg.livekit_observer_backend)
        cache_dir = cfg.livekit_cache_dir
        if cache_dir is None and cfg.workshop_dir is not None:
            cache_dir = Path(cfg.workshop_dir) / ".livekit_cache"
        if cache_dir is None:
            cache_dir = Path(".livekit_cache")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        worker = get_livekit_worker(cache_dir) if cfg.livekit_use_worker else None
        live_rows: list = []
        connected = 0
        joined_via_api = 0

        for stream in streams:
            credentials, via_api = _resolve_credentials(stream, cfg)
            if credentials is None:
                logger.warning(
                    "skipping live stream %s — no credentials and no fixture snapshots",
                    stream.stream_id,
                )
                continue

            rows = observer.observe(
                stream,
                credentials,
                cache_dir=cache_dir,
                capture_seconds=cfg.livekit_capture_seconds,
            )
            if rows:
                connected += 1
                live_rows.extend(rows)
                if worker is not None:
                    worker.append(
                        stream.stream_id,
                        rows,
                        title=stream.title,
                        curriculum_topic=stream.curriculum_topic,
                    )

            if via_api and cfg.social_api_base_url:
                joined_via_api += 1
                leave_live_stream(
                    str(cfg.social_api_base_url),
                    stream.stream_id,
                    cfg.livekit_researcher_identity,
                )

        if live_rows:
            existing = list(pipeline.multimodal_synced)
            existing.extend(live_rows)
            pipeline.multimodal_synced = existing

            for index, package in enumerate(pipeline.packages):
                if not package.multimodal_evidence:
                    continue
                pipeline.packages[index] = attach_multimodal_evidence(
                    package,
                    live_rows,
                    curriculum_topic=cfg.workshop_curriculum_topic,
                )

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "understanding.livestream: evidence=%d streams_connected=%d joined_via_api=%d",
            len(live_rows),
            connected,
            joined_via_api,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(live_rows),
                duration_ms=round(elapsed, 2),
                notes=f"streams={connected} api_joins={joined_via_api}",
            ),
            artifacts={"live_evidence": live_rows},
        )


class LiveKitArchiveCapability:
    """L2 — archive live evidence to offline timeline fixtures when streams end."""

    level = 2
    name = "understanding.livekit.archive"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        cfg = ctx.config
        if not cfg.enable_livekit or not cfg.enable_livekit_archive:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="livekit archive disabled",
                ),
            )

        archive_dir = cfg.livekit_archive_dir
        if archive_dir is None and cfg.livekit_cache_dir is not None:
            archive_dir = Path(cfg.livekit_cache_dir) / "archives"
        if archive_dir is None and cfg.workshop_dir is not None:
            archive_dir = Path(cfg.workshop_dir) / ".livekit_cache" / "archives"
        if archive_dir is None:
            archive_dir = Path(".livekit_cache/archives")
        archive_dir = Path(archive_dir)

        live_rows = [row for row in pipeline.multimodal_synced if row.is_live]
        stream_ids = {row.live_stream_id for row in live_rows if row.live_stream_id}
        for stream in pipeline.live_streams:
            stream_ids.add(stream.stream_id)

        archived_paths: list[str] = []
        for stream_id in sorted(stream_ids):
            if not stream_id:
                continue
            rows = [row for row in live_rows if row.live_stream_id == stream_id]
            title = stream_id
            topic = cfg.workshop_curriculum_topic
            for stream in pipeline.live_streams:
                if stream.stream_id == stream_id:
                    title = stream.title
                    topic = stream.curriculum_topic
                    break
            path = archive_evidence_rows(
                stream_id=stream_id,
                title=title,
                curriculum_topic=topic,
                evidence=rows,
                output_dir=archive_dir,
            )
            if path is not None:
                archived_paths.append(str(path))

        elapsed = (time.perf_counter() - started) * 1000
        logger.info("understanding.livekit.archive: archived=%d", len(archived_paths))
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(archived_paths),
                duration_ms=round(elapsed, 2),
                notes=f"dir={archive_dir}",
            ),
            artifacts={"archived_paths": archived_paths},
        )
