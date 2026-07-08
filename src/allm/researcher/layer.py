"""Researcher layer — capability-driven discover, package, recommend."""

from __future__ import annotations

import json
import os
from pathlib import Path

from allm.core.logging import get_logger
from allm.knowledge.graph import KnowledgeGraph
from allm.researcher.capabilities.base import CapabilityContext, ResearcherPipelineConfig
from allm.researcher.capabilities.improvement import ImprovementCapability
from allm.researcher.ecosystem_metrics import ResearcherEcosystemMetrics, compute_ecosystem_metrics
from allm.researcher.pipeline import CapabilityPipeline
from allm.researcher.queue import RecommendationQueue
from allm.researcher.types import ResearchRecommendation, ResearcherReport
from allm.storage.base import RecordStore
from allm.teacher.state import KnowledgeState

logger = get_logger("researcher")

HINTS_NAMESPACE = "researcher_strategy_hints"
HINTS_KEY = "latest"
KEL_REQUESTS_NAMESPACE = "kel_research_requests"


class ResearcherLayer:
    """Curiosity engine: external sources → packages → Teacher recommendations."""

    def __init__(
        self,
        store: RecordStore,
        *,
        workshop_dir: Path | str | None = None,
        software_samples: Path | str | None = None,
        repository_dir: Path | str | None = None,
        repository_max_files: int | None = 48,
        workshop_max_files: int | None = 3,
        workshop_curriculum_topic: str = "kids-plasma",
        book_dir: Path | str | None = None,
        book_max_files: int | None = 1,
        book_max_pages: int = 32,
        book_curriculum_topic: str = "kids-plasma",
        book_pdf_backend: str = "auto",
        enable_book_images: bool = False,
        book_max_images: int = 24,
        book_images_cache_dir: Path | str | None = None,
        catalog_topics: tuple[str, ...] | None = None,
        video_fixture_dir: Path | str | None = None,
        video_dir: Path | str | None = None,
        auto_generate_video_fixtures: bool = False,
        enable_vision_captions: bool = False,
        frames_cache_dir: Path | str | None = None,
        vision_caption_backend: str = "stub",
        vision_ollama_model: str | None = None,
        enable_audio_analysis: bool = False,
        audio_cache_dir: Path | str | None = None,
        audio_analysis_backend: str = "auto",
        enable_frame_ocr: bool = False,
        ocr_backend: str = "auto",
        ocr_ollama_model: str | None = None,
        enable_vision_analytics: bool = False,
        vision_analytics_backend: str = "auto",
        enable_motion_tracking: bool = False,
        motion_tracking_backend: str = "auto",
        motion_tracking_samples: int = 3,
        motion_tracking_fps: float = 30.0,
        enable_motion_continuity: bool = False,
        motion_continuity_min_score: float = 0.35,
        enable_object_identity: bool = False,
        object_identity_min_score: float = 0.30,
        enable_visual_distillation: bool = False,
        visual_distillation_max_images: int = 3,
        visual_distillation_max_questions: int = 5,
        enable_visual_export: bool = False,
        visual_export_auto_approve: bool = True,
        visual_export_min_confidence: float = 0.7,
        visual_export_max_images: int = 2,
        visual_export_max_questions: int = 3,
        visual_export_persist_approvals: bool = True,
        enable_cross_source_verification: bool = True,
        cross_source_min_overlap: float = 0.35,
        enable_livekit: bool = False,
        social_api_base_url: str | None = None,
        livekit_fixture_path: Path | str | None = None,
        livekit_stream_ids: tuple[str, ...] | None = None,
        livekit_topics: tuple[str, ...] | None = None,
        livekit_observer_backend: str = "auto",
        livekit_cache_dir: Path | str | None = None,
        livekit_capture_seconds: float = 3.0,
        livekit_use_worker: bool = False,
        enable_livekit_archive: bool = False,
        enabled_capabilities: tuple[str, ...] | None = None,
        graph: KnowledgeGraph | None = None,
        state: KnowledgeState | None = None,
        identities: dict | None = None,
        student_ids: tuple[str, ...] | None = None,
        kel_findings: tuple = (),
        ecosystem: ResearcherEcosystemMetrics | None = None,
    ) -> None:
        self._store = store
        self._queue = RecommendationQueue(store)
        self._pipeline = CapabilityPipeline(store, self._queue)
        self._workshop_dir = Path(workshop_dir) if workshop_dir else None
        self._software_samples = Path(software_samples) if software_samples else None
        self._repository_dir = Path(repository_dir) if repository_dir else None
        self._repository_max_files = repository_max_files
        self._workshop_max_files = workshop_max_files
        self._workshop_curriculum_topic = workshop_curriculum_topic
        self._book_dir = Path(book_dir) if book_dir else None
        self._book_max_files = book_max_files
        self._book_max_pages = book_max_pages
        self._book_curriculum_topic = book_curriculum_topic
        self._book_pdf_backend = book_pdf_backend or os.environ.get("ALLM_BOOK_PDF_BACKEND", "auto")
        self._enable_book_images = enable_book_images
        self._book_max_images = book_max_images
        self._book_images_cache_dir = Path(book_images_cache_dir) if book_images_cache_dir else None
        self._video_fixture_dir = Path(video_fixture_dir) if video_fixture_dir else None
        self._video_dir = Path(video_dir) if video_dir else None
        self._auto_generate_video_fixtures = auto_generate_video_fixtures
        self._enable_vision_captions = enable_vision_captions
        self._frames_cache_dir = Path(frames_cache_dir) if frames_cache_dir else None
        self._vision_caption_backend = vision_caption_backend
        self._vision_ollama_model = vision_ollama_model or os.environ.get(
            "ALLM_VISION_MODEL", "llava"
        )
        self._enable_audio_analysis = enable_audio_analysis
        self._audio_cache_dir = Path(audio_cache_dir) if audio_cache_dir else None
        self._audio_analysis_backend = audio_analysis_backend
        self._enable_frame_ocr = enable_frame_ocr
        self._ocr_backend = ocr_backend
        self._ocr_ollama_model = ocr_ollama_model or os.environ.get(
            "ALLM_OCR_MODEL",
            os.environ.get("ALLM_VISION_MODEL", "llava"),
        )
        self._enable_vision_analytics = enable_vision_analytics
        self._vision_analytics_backend = vision_analytics_backend or os.environ.get(
            "ALLM_VISION_ANALYTICS_BACKEND", "auto"
        )
        self._enable_motion_tracking = enable_motion_tracking
        self._motion_tracking_backend = motion_tracking_backend or os.environ.get(
            "ALLM_MOTION_TRACKING_BACKEND", "auto"
        )
        self._motion_tracking_samples = motion_tracking_samples
        self._motion_tracking_fps = motion_tracking_fps
        self._enable_motion_continuity = enable_motion_continuity
        self._motion_continuity_min_score = motion_continuity_min_score
        self._enable_object_identity = enable_object_identity
        self._object_identity_min_score = object_identity_min_score
        self._enable_visual_distillation = enable_visual_distillation
        self._visual_distillation_max_images = visual_distillation_max_images
        self._visual_distillation_max_questions = visual_distillation_max_questions
        self._enable_visual_export = enable_visual_export
        self._visual_export_auto_approve = visual_export_auto_approve
        self._visual_export_min_confidence = visual_export_min_confidence
        self._visual_export_max_images = visual_export_max_images
        self._visual_export_max_questions = visual_export_max_questions
        self._visual_export_persist_approvals = visual_export_persist_approvals
        self._enable_cross_source_verification = enable_cross_source_verification
        self._cross_source_min_overlap = cross_source_min_overlap
        if os.environ.get("ALLM_CROSS_SOURCE_VERIFY", "1") == "0":
            self._enable_cross_source_verification = False
        self._enable_livekit = enable_livekit
        self._social_api_base_url = social_api_base_url or os.environ.get("ALLM_SOCIAL_API_URL")
        self._livekit_fixture_path = Path(livekit_fixture_path) if livekit_fixture_path else None
        self._livekit_stream_ids = tuple(livekit_stream_ids or ())
        if self._livekit_stream_ids == () and os.environ.get("ALLM_LIVEKIT_STREAM_ID"):
            self._livekit_stream_ids = (os.environ["ALLM_LIVEKIT_STREAM_ID"],)
        self._livekit_topics = set(livekit_topics or ())
        self._livekit_observer_backend = livekit_observer_backend
        self._livekit_cache_dir = Path(livekit_cache_dir) if livekit_cache_dir else None
        self._livekit_capture_seconds = livekit_capture_seconds
        self._livekit_use_worker = livekit_use_worker
        self._enable_livekit_archive = enable_livekit_archive
        self._catalog_topics = set(catalog_topics or ())
        self._enabled_capabilities = enabled_capabilities
        self._graph = graph
        self._state = state
        self._identities = identities or {}
        self._student_ids = student_ids or ()
        self._kel_findings = kel_findings
        self._ecosystem = ecosystem
        self._last_report: ResearcherReport | None = None

    def run_cycle(self) -> ResearcherReport:
        """Discover sources, build packages, enqueue recommendations."""
        strategy_hints = self._load_strategy_hints()
        ctx = CapabilityContext(
            store=self._store,
            config=ResearcherPipelineConfig(
                workshop_dir=self._workshop_dir,
                software_samples=self._software_samples,
                repository_dir=self._repository_dir,
                repository_max_files=self._repository_max_files,
                workshop_max_files=self._workshop_max_files,
                workshop_curriculum_topic=self._workshop_curriculum_topic,
                book_dir=self._book_dir,
                book_max_files=self._book_max_files,
                book_max_pages=self._book_max_pages,
                book_curriculum_topic=self._book_curriculum_topic,
                book_pdf_backend=self._book_pdf_backend,
                enable_book_images=self._enable_book_images,
                book_max_images=self._book_max_images,
                book_images_cache_dir=self._book_images_cache_dir,
                video_fixture_dir=self._video_fixture_dir,
                video_dir=self._video_dir,
                auto_generate_video_fixtures=self._auto_generate_video_fixtures,
                enable_vision_captions=self._enable_vision_captions,
                frames_cache_dir=self._frames_cache_dir,
                vision_caption_backend=self._vision_caption_backend,
                vision_ollama_model=self._vision_ollama_model,
                enable_audio_analysis=self._enable_audio_analysis,
                audio_cache_dir=self._audio_cache_dir,
                audio_analysis_backend=self._audio_analysis_backend,
                enable_frame_ocr=self._enable_frame_ocr,
                ocr_backend=self._ocr_backend,
                ocr_ollama_model=self._ocr_ollama_model,
                enable_vision_analytics=self._enable_vision_analytics,
                vision_analytics_backend=self._vision_analytics_backend,
                enable_motion_tracking=self._enable_motion_tracking,
                motion_tracking_backend=self._motion_tracking_backend,
                motion_tracking_samples=self._motion_tracking_samples,
                motion_tracking_fps=self._motion_tracking_fps,
                enable_motion_continuity=self._enable_motion_continuity,
                motion_continuity_min_score=self._motion_continuity_min_score,
                enable_object_identity=self._enable_object_identity,
                object_identity_min_score=self._object_identity_min_score,
                enable_visual_distillation=self._enable_visual_distillation,
                visual_distillation_max_images=self._visual_distillation_max_images,
                visual_distillation_max_questions=self._visual_distillation_max_questions,
                enable_visual_export=self._enable_visual_export,
                visual_export_auto_approve=self._visual_export_auto_approve,
                visual_export_min_confidence=self._visual_export_min_confidence,
                visual_export_max_images=self._visual_export_max_images,
                visual_export_max_questions=self._visual_export_max_questions,
                visual_export_persist_approvals=self._visual_export_persist_approvals,
                enable_cross_source_verification=self._enable_cross_source_verification,
                cross_source_min_overlap=self._cross_source_min_overlap,
                enable_livekit=self._enable_livekit,
                social_api_base_url=self._social_api_base_url,
                livekit_fixture_path=self._livekit_fixture_path,
                livekit_stream_ids=self._livekit_stream_ids,
                livekit_topics=frozenset(self._livekit_topics),
                livekit_observer_backend=self._livekit_observer_backend,
                livekit_cache_dir=self._livekit_cache_dir,
                livekit_capture_seconds=self._livekit_capture_seconds,
                livekit_use_worker=self._livekit_use_worker,
                enable_livekit_archive=self._enable_livekit_archive,
                catalog_topics=frozenset(self._catalog_topics),
                enabled_capabilities=self._enabled_capabilities,
                discovery_source_order=os.environ.get("ALLM_DISCOVERY_ORDER") or None,
            ),
            graph=self._graph,
            state=self._state,
            identities=self._identities,
            student_ids=self._student_ids,
            kel_findings=self._kel_findings,
            ecosystem=self._ecosystem,
            strategy_hints=strategy_hints,
        )
        report = self._pipeline.run_cycle(ctx)
        if report.strategy_hints is not None:
            self._store.put(
                HINTS_NAMESPACE,
                HINTS_KEY,
                {"notes": list(getattr(report.strategy_hints, "notes", ()))},
                reason="improvement cycle",
            )
        self._last_report = report
        return report

    def set_catalog_topics(self, topics: set[str] | tuple[str, ...]) -> None:
        """Update curriculum keys used when aligning package concepts."""
        self._catalog_topics = set(topics)

    def set_context(
        self,
        *,
        graph: KnowledgeGraph | None = None,
        state: KnowledgeState | None = None,
        identities: dict | None = None,
        student_ids: tuple[str, ...] | None = None,
        kel_findings: tuple = (),
        ecosystem: ResearcherEcosystemMetrics | None = None,
    ) -> None:
        """Update runtime context for targeting and planning."""
        if graph is not None:
            self._graph = graph
        if state is not None:
            self._state = state
        if identities is not None:
            self._identities = identities
        if student_ids is not None:
            self._student_ids = student_ids
        if kel_findings:
            self._kel_findings = kel_findings
        if ecosystem is not None:
            self._ecosystem = ecosystem

    def active_recommendations(
        self,
        *,
        limit: int = 16,
        student_id: str | None = None,
    ) -> list[ResearchRecommendation]:
        """Recommendations waiting for Teacher/KEL, optionally filtered by student."""
        rows = self._queue.active(limit=None)
        if student_id is not None:
            rows = [
                rec
                for rec in rows
                if student_id not in rec.skip_students
                and (not rec.suggested_students or student_id in rec.suggested_students)
            ]
        if limit is None:
            return rows
        return rows[:limit]

    def ecosystem_metrics(
        self,
        graph: KnowledgeGraph,
        state: KnowledgeState,
        *,
        mastery_threshold: float = 0.75,
    ) -> ResearcherEcosystemMetrics:
        """Compute ecosystem metrics from stored packages and recommendations."""
        return compute_ecosystem_metrics(
            graph,
            state,
            self._queue.active(),
            self._queue.packages(),
            store=self._queue._store,
            mastery_threshold=mastery_threshold,
            student_ids=self._student_ids,
        )

    def last_report(self) -> ResearcherReport | None:
        """Most recent pipeline report."""
        return self._last_report

    def stored_packages(self) -> list:
        """All Knowledge Packages persisted from prior cycles."""
        return self._queue.packages()

    def persist_package(self, package, *, reason: str = "updated") -> None:
        """Store an updated Knowledge Package snapshot."""
        self._queue.store_package(package, reason=reason)

    def student_visual_packages(self, *, topic: str | None = None) -> tuple:
        """Teacher-approved student visual exports, optionally filtered by topic."""
        from allm.teacher.student_visual_delivery import packages_for_topic

        packages = tuple(self._queue.packages())
        if topic is None:
            exports = []
            seen: set[str] = set()
            for package in packages:
                for export in package.student_visual_packages:
                    if export.export_id not in seen:
                        seen.add(export.export_id)
                        exports.append(export)
            return tuple(exports)
        return packages_for_topic(packages, topic)

    def submit_kel_research_requests(
        self,
        requests: tuple,
        *,
        diagnostic_context=None,
    ) -> int:
        """Enqueue KEL remediation tasks for the Researcher to investigate."""
        import json

        from allm.kel.research_requests import KelResearchRequest
        from allm.researcher.curriculum_diagnostics import (
            DiagnosticContext,
            curriculum_diagnostics_enabled,
            diagnose_requests,
            format_diagnostic,
            reasoning_diagnostics_enabled,
        )
        from allm.researcher.model_router import resolve_model_spec, route_request
        from allm.researcher.remediation import requests_to_recommendations
        from allm.models import load_model

        typed = tuple(
            req if isinstance(req, KelResearchRequest) else KelResearchRequest.model_validate(req)
            for req in requests
        )
        if not typed:
            return 0
        for request in typed:
            self._store.put(
                KEL_REQUESTS_NAMESPACE,
                request.id,
                json.loads(request.model_dump_json()),
                reason="kel research request",
            )

        context = diagnostic_context
        if context is None:
            context = DiagnosticContext()
        elif not isinstance(context, DiagnosticContext):
            context = DiagnosticContext.model_validate(context)

        diagnostics = ()
        if curriculum_diagnostics_enabled():
            model = None
            if reasoning_diagnostics_enabled():
                role = route_request(typed[0])
                if role != "vision":
                    try:
                        model = load_model(resolve_model_spec(role))
                    except (OSError, RuntimeError):
                        model = None
            diagnostics = diagnose_requests(typed, context, model=model)
            for diagnostic in diagnostics:
                self._store.put(
                    "curriculum_diagnostics",
                    diagnostic.request_id,
                    json.loads(diagnostic.model_dump_json()),
                    reason="curriculum diagnostic",
                )
                logger.info("diagnostics.curriculum: %s", format_diagnostic(diagnostic))

        recommendations = requests_to_recommendations(typed, diagnostics=diagnostics)
        for recommendation in recommendations:
            self._queue.enqueue(recommendation, reason="kel_research_request")
        logger.info(
            "KEL research requests: enqueued %d remediation recommendation(s)",
            len(recommendations),
        )
        return len(recommendations)

    def kel_research_requests(self) -> list:
        """Persisted KEL research requests, newest first."""
        from allm.kel.research_requests import KelResearchRequest

        rows: list[KelResearchRequest] = []
        for key in self._store.keys(KEL_REQUESTS_NAMESPACE):
            record = self._store.get(KEL_REQUESTS_NAMESPACE, key)
            if record is None:
                continue
            rows.append(KelResearchRequest.model_validate(record.value))
        rows.sort(key=lambda row: -row.priority)
        return rows

    def evidence_broker(self):
        """Broker for debate/Teacher evidence retrieval from stored packages."""
        from allm.researcher.evidence_broker import EvidenceBroker

        return EvidenceBroker.from_store(self._store)

    def connect_livekit(self, stream_id: str, *, room_name: str | None = None):
        """Build LiveKit observer credentials for a live workshop stream."""
        from allm.researcher.livekit_tokens import credentials_for_stream, load_livekit_config
        from allm.researcher.livekit_types import LiveStreamInfo
        from allm.researcher.social_stream_client import join_live_stream

        identity = os.environ.get("ALLM_LIVEKIT_IDENTITY", "plasma-researcher")
        if self._social_api_base_url:
            joined = join_live_stream(
                str(self._social_api_base_url),
                stream_id,
                identity,
                role="viewer",
            )
            if joined is not None:
                _, credentials = joined
                return credentials

        config = load_livekit_config()
        if config is None:
            raise RuntimeError("LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are required")
        stream = LiveStreamInfo(
            stream_id=stream_id,
            title=stream_id,
            status="live",
            livekit_room_name=room_name or stream_id,
            livekit_url=config.url,
            curriculum_topic=self._workshop_curriculum_topic,
        )
        return credentials_for_stream(stream, config, identity=identity)

    def livekit_worker(self):
        """Return the persistent LiveKit evidence buffer."""
        from allm.researcher.livekit_worker import get_livekit_worker

        cache = self._livekit_cache_dir
        if cache is None and self._workshop_dir is not None:
            cache = self._workshop_dir / ".livekit_cache"
        if cache is None:
            cache = Path(".livekit_cache")
        return get_livekit_worker(cache)

    def _load_strategy_hints(self):
        return ImprovementCapability.load_strategy_hints(self._store)
