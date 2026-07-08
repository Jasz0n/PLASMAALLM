"""Dual-specialist loop with Teacher-mediated consultation (offline-friendly)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from allm.collector import SamplePool
from allm.data.base import Sample
from allm.debate import DebateEngine
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.mixed_corpus import load_mixed_corpus
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import LearningLoop, LoopConfig
from allm.loop.debate_evidence import DebateEvidenceSummary
from allm.memory import EpisodicMemory
from allm.models import EchoModel, ModelSpec
from allm.planner import NeedPlanner
from allm.students import FailureLog, ModelStudent, ModelStudentConfig
from allm.students.identity import StudentIdentity
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import InContextTrainer

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DualConsultRunResult:
    """Outcome of a dual-student mediated consultation loop."""

    software_score_last: float
    plasma_score_last: float
    mediated_approvals: int
    debate_evidence_hits: int
    researcher_packages: int
    multimodal_synced: int
    live_evidence_count: int
    archived_fixtures: int
    worker_streams: int
    workdir: Path


def _make_student(student_id: str, topic: str, train: list[Sample]) -> ModelStudent:
    student = ModelStudent(
        student_id,
        topic,
        EchoModel(ModelSpec(name=student_id, provider="echo", model_id="none")),
        ModelStudentConfig(max_notes=max(len(train), 128), notes_in_prompt=16),
    )
    InContextTrainer().train(student, train)
    return student


def run_dual_mediated_loop(
    *,
    plasma_identity: StudentIdentity,
    software_identity: StudentIdentity,
    dry_run: bool = True,
    workdir: Path | str | None = None,
    verbose: bool = True,
) -> DualConsultRunResult:
    """Run software + plasma specialists in one loop with mediated consultation."""
    corpus = load_mixed_corpus(ROOT)
    iterations = int(os.environ.get("ALLM_ITERATIONS", "2"))
    loop_seed = int(os.environ.get("ALLM_LOOP_SEED", "42"))
    use_researcher = os.environ.get("ALLM_RESEARCHER", "0") == "1"
    use_multimodal = os.environ.get("ALLM_MULTIMODAL", "0") == "1"
    use_debate_evidence = os.environ.get("ALLM_DEBATE_EVIDENCE", "0") == "1"
    use_consult_show_me = os.environ.get("ALLM_CONSULT_SHOW_ME", "0") == "1"
    use_vision = os.environ.get("ALLM_VISION_CAPTIONS", "0") == "1"
    use_audio = os.environ.get("ALLM_AUDIO_ANALYSIS", "0") == "1"
    use_ocr = os.environ.get("ALLM_FRAME_OCR", "0") == "1"
    use_vision_analytics = os.environ.get("ALLM_VISION_ANALYTICS", "0") == "1"
    use_motion_tracking = os.environ.get("ALLM_MOTION_TRACKING", "0") == "1"
    use_motion_continuity = os.environ.get("ALLM_MOTION_CONTINUITY", "0") == "1"
    use_object_identity = os.environ.get("ALLM_OBJECT_IDENTITY", "0") == "1"
    use_visual_distill = os.environ.get("ALLM_VISUAL_DISTILL", "0") == "1"
    use_visual_export = os.environ.get("ALLM_VISUAL_EXPORT", "0") == "1"
    use_livekit = os.environ.get("ALLM_LIVEKIT", "0") == "1"
    use_livekit_archive = os.environ.get("ALLM_LIVEKIT_ARCHIVE", "0") == "1"
    use_livekit_worker = os.environ.get("ALLM_LIVEKIT_WORKER", "0") == "1"

    run_dir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="allm-dual-consult-"))
    livekit_cache = run_dir / "livekit"
    store = SQLiteRecordStore(run_dir / "state.sqlite3")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="kids-plasma", description="Kids plasma science"))
    graph.add(Concept(name="fastify-api", description="Software APIs"))

    plasma_holdout = [s for s in corpus.plasma.holdout if s.target][:2]
    software_holdout = [s for s in corpus.software.holdout if s.target][:2]
    cross_holdout = plasma_holdout + software_holdout
    teacher = Teacher(
        KnowledgeState(store),
        DatasetExamGenerator(cross_holdout),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )

    pool = SamplePool()
    pool.ingest(corpus.merged_train)

    if dry_run:
        plasma_student = _make_student(
            plasma_identity.student_id,
            "kids-plasma",
            list(corpus.plasma.train),
        )
        software_student = _make_student(
            software_identity.student_id,
            "fastify-api",
            list(corpus.software.train),
        )
    else:
        plasma_student = ModelStudent(
            plasma_identity.student_id,
            "kids-plasma",
            EchoModel(ModelSpec(name="plasma", provider="echo", model_id="none")),
            ModelStudentConfig(max_notes=128, notes_in_prompt=12),
        )
        software_student = ModelStudent(
            software_identity.student_id,
            "fastify-api",
            EchoModel(ModelSpec(name="software", provider="echo", model_id="none")),
            ModelStudentConfig(max_notes=128, notes_in_prompt=12),
        )
        trainer = InContextTrainer()
        trainer.train(plasma_student, list(corpus.plasma.train))
        trainer.train(software_student, list(corpus.software.train))

    teacher.evaluate(
        plasma_student,
        DatasetExamGenerator(list(corpus.plasma.train[:8])).generate(num_questions=2, seed=1),
    )
    teacher.evaluate(
        software_student,
        DatasetExamGenerator(list(corpus.software.train[:8])).generate(num_questions=2, seed=2),
    )

    researcher = None
    researcher_packages = 0
    multimodal_synced = 0
    live_evidence_count = 0
    archived_fixtures = 0
    worker_streams = 0
    if use_researcher:
        from allm.researcher import ResearcherLayer

        fixture_dir = ROOT / "transcripts/Kids/visual" if use_multimodal else None
        researcher = ResearcherLayer(
            store,
            workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
            workshop_max_files=3,
            catalog_topics=("kids-plasma", "fastify-api"),
            video_fixture_dir=fixture_dir,
            auto_generate_video_fixtures=use_multimodal
            and os.environ.get("ALLM_AUTO_VIDEO_FIXTURES", "0") == "1",
            enable_vision_captions=use_vision,
            enable_audio_analysis=use_audio,
            enable_frame_ocr=use_ocr,
            enable_vision_analytics=use_vision_analytics,
            vision_analytics_backend=os.environ.get("ALLM_VISION_ANALYTICS_BACKEND", "auto"),
            enable_motion_tracking=use_motion_tracking,
            motion_tracking_backend=os.environ.get("ALLM_MOTION_TRACKING_BACKEND", "auto"),
            motion_tracking_samples=int(os.environ.get("ALLM_MOTION_TRACKING_SAMPLES", "3")),
            enable_motion_continuity=use_motion_continuity,
            motion_continuity_min_score=float(
                os.environ.get("ALLM_MOTION_CONTINUITY_MIN_SCORE", "0.35")
            ),
            enable_object_identity=use_object_identity,
            object_identity_min_score=float(
                os.environ.get("ALLM_OBJECT_IDENTITY_MIN_SCORE", "0.30")
            ),
            enable_visual_distillation=use_visual_distill,
            visual_distillation_max_images=int(os.environ.get("ALLM_VISUAL_DISTILL_IMAGES", "3")),
            visual_distillation_max_questions=int(os.environ.get("ALLM_VISUAL_DISTILL_QUESTIONS", "5")),
            enable_visual_export=use_visual_export,
            visual_export_auto_approve=os.environ.get("ALLM_VISUAL_EXPORT_AUTO", "1") == "1",
            visual_export_persist_approvals=os.environ.get("ALLM_VISUAL_EXPORT_PERSIST", "1") == "1",
            enable_livekit=use_livekit,
            enable_livekit_archive=use_livekit_archive,
            livekit_use_worker=use_livekit_worker,
            livekit_cache_dir=livekit_cache if (use_livekit or use_livekit_archive) else None,
            livekit_fixture_path=(fixture_dir / "livekit_streams_fixture.json")
            if fixture_dir and (fixture_dir / "livekit_streams_fixture.json").is_file()
            else None,
            social_api_base_url=os.environ.get("ALLM_SOCIAL_API_URL"),
            livekit_topics=("kids-plasma", "fastify-api"),
            graph=graph,
            state=teacher.state,
            student_ids=(plasma_identity.student_id, software_identity.student_id),
        )
        report = researcher.run_cycle()
        researcher_packages = len(report.packages)
        multimodal_synced = len(report.multimodal_synced)
        live_evidence_count = sum(1 for row in report.multimodal_synced if row.is_live)
        if use_livekit_worker and hasattr(researcher, "livekit_worker"):
            worker_streams = len(researcher.livekit_worker().stream_ids())
        archive_dir = livekit_cache / "archives"
        if archive_dir.is_dir():
            archived_fixtures = len(list(archive_dir.glob("*_archive.json")))
        if verbose:
            print("\n=== Researcher (pre-loop) ===")
            print(f"  packages: {researcher_packages}")
            print(f"  multimodal synced: {multimodal_synced}")
            if use_livekit:
                print(f"  live evidence: {live_evidence_count}")
            if use_livekit_worker:
                print(f"  worker streams: {worker_streams}")
            if use_livekit_archive:
                print(f"  archived fixtures: {archived_fixtures}")
            for name, yield_count, notes in report.capability_summary:
                if any(
                    token in name
                    for token in (
                        "vision",
                        "audio",
                        "ocr",
                        "vision.analytics",
                        "vision.motion",
                        "vision.continuity",
                        "vision.identity",
                        "visual.distill",
                        "visual.export",
                        "livekit",
                        "livestream",
                        "sync",
                    )
                ):
                    print(f"  capability: {name} yield={yield_count} ({notes})")

    memory = EpisodicMemory(store)
    loop = LearningLoop(
        teacher=teacher,
        students=[software_student, plasma_student],
        planner=NeedPlanner(),
        trainer=InContextTrainer(),
        pool=pool,
        memory=memory,
        failure_log=FailureLog(store),
        graph=graph,
        identities={
            plasma_identity.student_id: plasma_identity,
            software_identity.student_id: software_identity,
        },
        enable_mediated_consultation=os.environ.get("ALLM_MEDIATED_CONSULT", "1") == "1",
        debate=DebateEngine(disagreement_threshold=0.3) if use_debate_evidence else None,
        researcher=researcher,
        config=LoopConfig(
            iterations=iterations,
            questions_per_exam=min(4, len(cross_holdout)),
            samples_per_iteration=12,
            study_failures=False,
            seed=loop_seed,
            enable_debate_evidence=use_debate_evidence,
            enable_consult_show_me=use_consult_show_me and use_researcher,
            enable_visual_delivery=os.environ.get("ALLM_VISUAL_DELIVERY", "0") == "1",
        ),
    )

    if verbose:
        print("\n=== Dual specialist mediated consultation loop ===")
        print(f"  students: {software_identity.student_id}, {plasma_identity.student_id}")
        print(f"  mediated: {os.environ.get('ALLM_MEDIATED_CONSULT', '1') == '1'}")
        print(f"  debate evidence: {use_debate_evidence}")
        print(f"  consult show me: {use_consult_show_me}")

    software_last = plasma_last = 0.0
    debate_evidence_hits = 0
    reports = loop.run()
    for report in reports:
        for row in report.students:
            if row.student_id == software_identity.student_id:
                software_last = row.score_after
            if row.student_id == plasma_identity.student_id:
                plasma_last = row.score_after
            if verbose:
                print(
                    f"  iter {report.iteration} {row.student_id}: "
                    f"{row.score_before:.2f}->{row.score_after:.2f} studied={row.samples_studied}"
                )
        evidence = report.debate_evidence
        if isinstance(evidence, DebateEvidenceSummary) and evidence.found:
            debate_evidence_hits += evidence.hit_count
            if verbose:
                print(f"    debate evidence: {evidence.hit_count} hits conf={evidence.confidence:.2f}")

    mediated_approvals = sum(
        1
        for episode in memory.recall(actor=software_identity.student_id, kind="observation")
        if "mediated consult" in episode.summary and "approved" in episode.summary
    )

    store.close()
    return DualConsultRunResult(
        software_score_last=software_last,
        plasma_score_last=plasma_last,
        mediated_approvals=mediated_approvals,
        debate_evidence_hits=debate_evidence_hits,
        researcher_packages=researcher_packages,
        multimodal_synced=multimodal_synced,
        live_evidence_count=live_evidence_count,
        archived_fixtures=archived_fixtures,
        worker_streams=worker_streams,
        workdir=run_dir,
    )
