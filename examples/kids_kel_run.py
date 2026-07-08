"""Shared runner for Kids KEL-steered held-out loops."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from allm.collector import SamplePool
from allm.evaluation import (
    compute_marginal_strategy_gains,
    diagnose_holdout_gap,
    evaluate_student,
    export_strategy_phase_gains,
    format_holdout_gap_report,
    format_strategy_gain_report,
)
from allm.exam import (
    EscalatingGrader,
    ExactMatchGrader,
    LLMJudgeGrader,
    MultiDimensionalGrader,
    ParaphraseExamGenerator,
    multi_judge_enabled,
)
from allm.kdp.curriculum import load_curriculum_splits
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import KelSteeredLearningLoop, KelSteeringConfig, LearningRunManifest, LoopConfig
from allm.loop.phased_learning import discovery_order_from_phase, parse_learning_phases
from allm.memory import EpisodicMemory
from allm.models import ModelSpec, load_model
from allm.planner import NeedPlanner
from allm.students import FailureLog, ModelStudent, ModelStudentConfig
from allm.students.identity import StudentIdentity, load_identity
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import InContextTrainer

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class KidsKelRunResult:
    """Artifacts and scores from one KEL-steered kids run."""

    workdir: Path
    first_score: float
    last_score: float
    iterations_completed: int
    peak_score: float
    mission_enabled: bool
    identity_path: str | None
    loop_seed: int
    history_path: Path
    kel_lg: float | None
    kel_ks: float | None = None
    workshop_packages: int = 0
    book_packages: int = 0
    aligned_concepts: int = 0
    book_figures: int = 0
    student_visual_exports: int = 0
    visual_notes_delivered: int = 0
    teacher_approved_briefs: int = 0
    teacher_rejected_briefs: int = 0
    teacher_paused: bool = False
    multimodal_synced: int = 0
    learning_phase_order: str | None = None

    @property
    def heldout_gain(self) -> float:
        """Held-out exam score change across the loop."""
        return self.last_score - self.first_score


def student_spec() -> ModelSpec:
    size = os.environ.get("ALLM_STUDENT_SIZE", "large").lower()
    config_name = "ollama_student.yaml" if size == "small" else "ollama_student_7b.yaml"
    default_model = "qwen2.5:0.5b-instruct" if size == "small" else "qwen2.5:7b-instruct"
    model_id = os.environ.get("ALLM_STUDENT_MODEL", default_model)
    spec = ModelSpec.from_yaml(ROOT / "configs/models" / config_name)
    return spec.model_copy(update={"model_id": model_id})


def grader_spec() -> ModelSpec:
    use_cloud = os.environ.get("ALLM_GRADER", "local").lower() == "cloud"
    name = "ollama_grader_cloud.yaml" if use_cloud else "ollama_grader_local.yaml"
    return ModelSpec.from_yaml(ROOT / "configs/models" / name)


def steering_config() -> KelSteeringConfig:
    return KelSteeringConfig(
        lg_window=int(os.environ.get("ALLM_KEL_LG_WINDOW", "3")),
        mastery_threshold=float(os.environ.get("ALLM_KEL_MASTERY", "0.75")),
        strategy_advance_threshold=float(os.environ.get("ALLM_KEL_STRATEGY_MASTERY", "0.35")),
        strategy_advance_window=int(os.environ.get("ALLM_KEL_STRATEGY_WINDOW", "3")),
        stagnation_iterations=int(os.environ.get("ALLM_KEL_STAGNATION_ITERS", "3")),
        max_questions=int(os.environ.get("ALLM_KEL_MAX_QUESTIONS", "16")),
        max_samples=int(os.environ.get("ALLM_KEL_MAX_SAMPLES", "128")),
        sample_boost=float(os.environ.get("ALLM_KEL_SAMPLE_BOOST", "1.5")),
        question_boost=int(os.environ.get("ALLM_KEL_QUESTION_BOOST", "2")),
        min_iterations_before_halt=int(os.environ.get("ALLM_KEL_MIN_ITER_HALT", "6")),
        min_lg_history_for_halt=int(os.environ.get("ALLM_KEL_MIN_LG_HALT", "5")),
        halt_on_static_illusion=os.environ.get("ALLM_KEL_HALT_STATIC", "1") == "1",
        require_retention_stable=os.environ.get("ALLM_KEL_RETENTION_GATE", "1") == "1",
        retention_max_drop_from_peak=float(os.environ.get("ALLM_KEL_RETENTION_MAX_DROP", "0.15")),
        block_advance_on_forgetting=os.environ.get("ALLM_KEL_BLOCK_FORGETTING", "1") == "1",
        cap_samples_when_unstable=os.environ.get("ALLM_KEL_CAP_SAMPLES", "1") == "1",
    )


def resolve_identity(
    student_id: str,
    *,
    identity_path: str | None,
) -> tuple[dict[str, StudentIdentity], str | None, bool]:
    """Return identities dict, resolved path, and whether mission is enabled."""
    if identity_path is None or identity_path.lower() in {"0", "none", "off", "false"}:
        return {}, None, False

    identity_file = ROOT / identity_path if not Path(identity_path).is_absolute() else Path(identity_path)
    if not identity_file.is_file():
        return {}, None, False

    identity = load_identity(identity_file).model_copy(update={"student_id": student_id})
    return {student_id: identity}, str(identity_file), True


def run_kids_kel_steered(
    *,
    identity_path: str | None = "configs/students/kids_kel_plasma.yaml",
    workdir: Path | str | None = None,
    verbose: bool = True,
) -> KidsKelRunResult:
    """Run the Kids KEL-steered held-out loop and return summary metrics."""
    from allm.kdp.book_curriculum import apply_books_only_defaults, books_only_mode, load_test_curriculum_splits

    apply_books_only_defaults()
    os.environ.setdefault("ALLM_SAMPLES", "exam")
    train, holdout = load_test_curriculum_splits(ROOT)
    if len(train) < 4 or len(holdout) < 4:
        raise SystemExit(f"Need train and holdout pools (got {len(train)}/{len(holdout)})")

    iterations = int(os.environ.get("ALLM_ITERATIONS", "8"))
    questions = int(os.environ.get("ALLM_QUESTIONS", "8"))
    samples_per_iter = int(os.environ.get("ALLM_SAMPLES_PER_ITER", "32"))
    loop_seed = int(os.environ.get("ALLM_LOOP_SEED", "42"))
    max_notes = int(os.environ.get("ALLM_MAX_NOTES", str(max(len(train), 256))))
    notes_in_prompt = int(os.environ.get("ALLM_NOTES_IN_PROMPT", "16"))

    run_dir = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="allm-kids-kel-"))
    os.environ.setdefault("ALLM_JUDGE_DISAGREEMENT_LOG", str(run_dir / "judge_disagreements.jsonl"))
    store = SQLiteRecordStore(run_dir / "kids.sqlite3")
    state = KnowledgeState(store)
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC, description="Kids plasma science"))
    kel = KnowledgeEvaluationLayer(graph, store, state)

    judge_model = load_model(grader_spec())
    if multi_judge_enabled():
        judge = MultiDimensionalGrader(judge_model, compare_exact=True)
    else:
        judge = LLMJudgeGrader(judge_model, compare_exact=True)
    grader = EscalatingGrader(ExactMatchGrader("contains"), judge)
    teacher = Teacher(
        state,
        ParaphraseExamGenerator(holdout, paraphrase=False),
        grader,
        TeacherConfig(confidence_smoothing=0.5),
    )

    pool = SamplePool()
    pool.ingest(train)
    student = ModelStudent(
        "kids-kel",
        DEFAULT_TOPIC,
        load_model(student_spec()),
        ModelStudentConfig(max_notes=max_notes, notes_in_prompt=notes_in_prompt),
    )
    trainer = InContextTrainer()
    if os.environ.get("ALLM_BOOTSTRAP", "0") == "1":
        trainer.train(student, train)

    steering = steering_config()
    gap = diagnose_holdout_gap(train, holdout)
    identities, resolved_identity, mission_enabled = resolve_identity(
        student.student_id,
        identity_path=identity_path,
    )

    researcher = None
    research_report = None
    workshop_packages = book_packages = aligned_concepts = 0
    book_figures = student_visual_exports = multimodal_synced_count = 0
    teacher_approved_briefs = teacher_rejected_briefs = 0
    teacher_paused = False
    if os.environ.get("ALLM_RESEARCHER", "0") == "1":
        from allm.researcher import ResearcherLayer

        video_fixture_dir = None
        auto_generate = False
        if os.environ.get("ALLM_MULTIMODAL", "0") == "1":
            video_fixture_dir = ROOT / "transcripts/Kids/visual"
            auto_generate = os.environ.get("ALLM_AUTO_VIDEO_FIXTURES", "0") == "1"
        enable_vision = os.environ.get("ALLM_VISION_CAPTIONS", "0") == "1"
        enable_audio = os.environ.get("ALLM_AUDIO_ANALYSIS", "0") == "1"
        enable_ocr = os.environ.get("ALLM_FRAME_OCR", "0") == "1"
        enable_vision_analytics = os.environ.get("ALLM_VISION_ANALYTICS", "0") == "1"
        enable_motion_tracking = os.environ.get("ALLM_MOTION_TRACKING", "0") == "1"
        enable_motion_continuity = os.environ.get("ALLM_MOTION_CONTINUITY", "0") == "1"
        enable_object_identity = os.environ.get("ALLM_OBJECT_IDENTITY", "0") == "1"
        enable_visual_distillation = os.environ.get("ALLM_VISUAL_DISTILL", "0") == "1"
        teacher_ui_pause = os.environ.get("ALLM_TEACHER_UI_PAUSE", "0") == "1"
        teacher_ui_approval = teacher_ui_pause or os.environ.get("ALLM_TEACHER_UI_APPROVAL", "0") == "1"
        if teacher_ui_approval:
            enable_visual_export = False
        elif os.environ.get("ALLM_VISUAL_EXPORT", "0") == "1":
            enable_visual_export = True
        else:
            enable_visual_export = False
        enable_livekit = os.environ.get("ALLM_LIVEKIT", "0") == "1"
        enable_livekit_archive = os.environ.get("ALLM_LIVEKIT_ARCHIVE", "0") == "1"
        livekit_use_worker = os.environ.get("ALLM_LIVEKIT_WORKER", "0") == "1"
        book_dir = None
        if os.environ.get("ALLM_BOOK_DISCOVERY", "0") == "1":
            book_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
            if os.environ.get("ALLM_BOOK_BOOTSTRAP_SIDECARS", "auto").lower() in {"1", "true", "auto"}:
                from allm.researcher.book_corpus import (
                    audit_book_corpus,
                    bootstrap_book_corpus,
                    corpus_is_complete,
                )

                max_files = int(os.environ.get("ALLM_BOOK_MAX_FILES", "3"))
                entries = audit_book_corpus(book_dir, max_files=max_files)
                bootstrap_mode = os.environ.get("ALLM_BOOK_BOOTSTRAP_SIDECARS", "auto").lower()
                needs_bootstrap = bootstrap_mode in {"1", "true"} or not corpus_is_complete(entries)
                if needs_bootstrap:
                    template_dir = Path(
                        os.environ.get(
                            "ALLM_BOOK_SIDECAR_TEMPLATES",
                            str(ROOT / "books" / "sidecar_templates"),
                        )
                    )
                    bootstrap = bootstrap_book_corpus(
                        book_dir,
                        template_dir=template_dir,
                        max_files=max_files,
                    )
                    if verbose and bootstrap.created:
                        print("\n=== Book sidecar bootstrap ===")
                        print(f"  created: {', '.join(bootstrap.created)}")
                elif verbose:
                    print("\n=== Book corpus complete (3/3 readable, pages verified) ===")

        disc_order = discovery_order_from_phase()
        if disc_order:
            os.environ.setdefault("ALLM_DISCOVERY_ORDER", disc_order)

        researcher = ResearcherLayer(
            store,
            workshop_dir=None
            if books_only_mode()
            else ROOT / "transcripts/Kids/cleaned/mk",
            software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
            workshop_max_files=int(os.environ.get("ALLM_RESEARCHER_WORKSHOP_FILES", "0" if books_only_mode() else "3")),
            book_dir=book_dir,
            book_max_files=int(os.environ.get("ALLM_BOOK_MAX_FILES", "1")),
            book_max_pages=int(os.environ.get("ALLM_BOOK_MAX_PAGES", "32")),
            book_max_images=int(os.environ.get("ALLM_BOOK_MAX_IMAGES", "24")),
            book_pdf_backend=os.environ.get("ALLM_BOOK_PDF_BACKEND", "auto"),
            enable_book_images=os.environ.get("ALLM_BOOK_IMAGES", "0") == "1",
            book_images_cache_dir=Path(os.environ["ALLM_BOOK_IMAGES_CACHE"])
            if os.environ.get("ALLM_BOOK_IMAGES_CACHE")
            else None,
            catalog_topics=(DEFAULT_TOPIC,),
            video_fixture_dir=video_fixture_dir,
            video_dir=ROOT / "transcripts/Kids/videos" if (ROOT / "transcripts/Kids/videos").is_dir() else None,
            auto_generate_video_fixtures=auto_generate,
            enable_vision_captions=enable_vision,
            vision_caption_backend=os.environ.get("ALLM_VISION_BACKEND", "auto"),
            vision_ollama_model=os.environ.get("ALLM_VISION_MODEL", "llava"),
            enable_audio_analysis=enable_audio,
            enable_frame_ocr=enable_ocr,
            ocr_backend=os.environ.get("ALLM_OCR_BACKEND", "auto"),
            ocr_ollama_model=os.environ.get(
                "ALLM_OCR_MODEL",
                os.environ.get("ALLM_VISION_MODEL", "llava"),
            ),
            enable_vision_analytics=enable_vision_analytics,
            vision_analytics_backend=os.environ.get("ALLM_VISION_ANALYTICS_BACKEND", "auto"),
            enable_motion_tracking=enable_motion_tracking,
            motion_tracking_backend=os.environ.get("ALLM_MOTION_TRACKING_BACKEND", "auto"),
            enable_motion_continuity=enable_motion_continuity,
            motion_continuity_min_score=float(
                os.environ.get("ALLM_MOTION_CONTINUITY_MIN_SCORE", "0.35")
            ),
            enable_object_identity=enable_object_identity,
            object_identity_min_score=float(
                os.environ.get("ALLM_OBJECT_IDENTITY_MIN_SCORE", "0.30")
            ),
            enable_visual_distillation=enable_visual_distillation,
            enable_visual_export=enable_visual_export,
            visual_export_auto_approve=os.environ.get("ALLM_VISUAL_EXPORT_AUTO", "1") == "1",
            visual_export_persist_approvals=os.environ.get("ALLM_VISUAL_EXPORT_PERSIST", "1") == "1",
            enable_livekit=enable_livekit,
            enable_livekit_archive=enable_livekit_archive,
            livekit_use_worker=livekit_use_worker,
            livekit_fixture_path=(video_fixture_dir / "livekit_streams_fixture.json")
            if video_fixture_dir and (video_fixture_dir / "livekit_streams_fixture.json").is_file()
            else None,
            social_api_base_url=os.environ.get("ALLM_SOCIAL_API_URL"),
            livekit_topics=(DEFAULT_TOPIC,),
            frames_cache_dir=run_dir / "frames" if video_fixture_dir else None,
            graph=graph,
            state=state,
            student_ids=(student.student_id,),
        )
        research_report = researcher.run_cycle()
        if teacher_ui_approval:
            if teacher_ui_pause:
                from allm.teacher.teacher_kel_session import (
                    TeacherKelSessionStore,
                    finalize_teacher_export,
                    resume_file_path,
                    wait_for_teacher_export,
                )
                from allm.teacher.visual_review_service import TeacherVisualReviewService

                teacher_paused = True
                review = TeacherVisualReviewService(store, packages=list(research_report.packages))
                review_summary = review.summary()
                TeacherKelSessionStore(store).open(
                    run_dir,
                    pending_briefs=review_summary["pending"],
                    total_briefs=review_summary["total_briefs"],
                )
                flag = resume_file_path(
                    run_dir,
                    override=os.environ.get("ALLM_TEACHER_RESUME_FILE"),
                )
                if verbose:
                    print("\n=== Teacher UI PAUSE (awaiting review) ===")
                    print(f"  pending briefs: {review_summary['pending']}")
                    print(f"  review UI: GET /teacher/")
                    print(f"  resume flag: {flag}")
                timeout = os.environ.get("ALLM_TEACHER_PAUSE_TIMEOUT")
                wait_for_teacher_export(
                    store,
                    resume_file=flag,
                    timeout_sec=float(timeout) if timeout else None,
                    poll_sec=float(os.environ.get("ALLM_TEACHER_PAUSE_POLL", "2")),
                )
                bridge_result = finalize_teacher_export(
                    store,
                    researcher,
                    research_report.packages,
                )
                TeacherKelSessionStore(store).mark_resumed()
            else:
                from allm.teacher.visual_kel_bridge import (
                    export_teacher_approved,
                    policy_from_env,
                    sync_researcher_packages,
                )

                bridge_result = export_teacher_approved(
                    store,
                    research_report.packages,
                    policy=policy_from_env(),
                )
                sync_researcher_packages(researcher, bridge_result.packages)
            teacher_approved_briefs = bridge_result.approved_count
            teacher_rejected_briefs = bridge_result.rejected_count
            student_visual_exports = len(bridge_result.exports)
            if verbose:
                label = "pause export" if teacher_ui_pause else "selective export"
                print(f"\n=== Teacher UI {label} (pre-loop) ===")
                print(f"  approved briefs: {teacher_approved_briefs}")
                print(f"  rejected briefs: {teacher_rejected_briefs}")
                print(f"  student visual exports: {student_visual_exports}")
        workshop_packages = sum(1 for pkg in research_report.packages if pkg.provider == "kids-workshops")
        book_packages = sum(1 for pkg in research_report.packages if pkg.provider == "keshe-books")
        multimodal_synced_count = len(research_report.multimodal_synced)
        book_figures = sum(
            1 for row in research_report.multimodal_synced if str(row.source_id).startswith("book:")
        )
        if not teacher_ui_approval:
            student_visual_exports = sum(
                len(pkg.student_visual_packages) for pkg in research_report.packages
            )
        cross_report = research_report.cross_source_report
        if cross_report is not None:
            aligned_concepts = getattr(cross_report, "aligned_count", 0)
        if verbose:
            print("\n=== Researcher cycle (pre-loop) ===")
            print(f"  packages: {len(research_report.packages)}")
            print(f"  workshop packages: {workshop_packages}")
            print(f"  book packages: {book_packages}")
            print(f"  recommendations: {len(research_report.recommendations)}")
            print(f"  conflicts preserved: {research_report.conflicts_detected}")
            print(f"  multimodal synced: {multimodal_synced_count}")
            print(f"  book figures: {book_figures}")
            print(f"  student visual exports: {student_visual_exports}")
            if cross_report is not None:
                print(f"  cross-source aligned: {aligned_concepts}")
                for alignment in getattr(cross_report, "alignments", ())[:5]:
                    print(
                        f"    ↔ {alignment.workshop_concept[:40]} "
                        f"/ {alignment.book_concept[:40]} "
                        f"({alignment.overlap_score:.2f})"
                    )
            for name, yield_count, notes in research_report.capability_summary:
                if name.startswith(("discovery.", "understanding.", "verification.")):
                    print(f"  capability: {name} yield={yield_count} ({notes})")

    manifest = LearningRunManifest(
        student_model=student_spec().model_id,
        train_count=gap.train_count,
        holdout_count=gap.holdout_count,
        holdout_exact_prompt_matches=gap.exact_prompt_matches,
        holdout_high_overlap=gap.high_overlap,
        holdout_low_overlap=gap.low_overlap,
        holdout_novel_lexical=gap.novel_lexical,
        holdout_answers_in_train=gap.answers_in_train,
        holdout_by_workshop=gap.by_workshop,
        kel_mastery_threshold=steering.mastery_threshold,
        kel_strategy_advance_threshold=steering.strategy_advance_threshold,
    )

    memory = EpisodicMemory(store)
    forgetting_watchdog = None
    if os.environ.get("ALLM_FORGETTING_WATCHDOG", "1") == "1":
        from allm.trainer import ForgettingWatchdog

        forgetting_watchdog = ForgettingWatchdog(
            teacher,
            mastery_threshold=float(os.environ.get("ALLM_FORGETTING_MASTERY", "0.25")),
            regression_threshold=float(os.environ.get("ALLM_FORGETTING_REGRESSION", "0.10")),
        )
    phases = parse_learning_phases()
    phase_order_key = os.environ.get("ALLM_KEL_PHASE_ORDER", "").strip().lower() or None
    loop_iterations = (
        sum(phase.iterations for phase in phases) if phases else iterations
    )
    loop = KelSteeredLearningLoop(
        kel=kel,
        steering=steering,
        history_path=run_dir / "learning_history.jsonl",
        holdout_gap=gap,
        run_manifest=manifest,
        identities=identities or None,
        researcher=researcher,
        teacher=teacher,
        students=[student],
        planner=NeedPlanner(),
        trainer=trainer,
        pool=pool,
        memory=memory,
        failure_log=FailureLog(store),
        graph=graph,
        forgetting=forgetting_watchdog,
        config=LoopConfig(
            iterations=loop_iterations,
            questions_per_exam=min(questions, len(holdout)),
            samples_per_iteration=min(samples_per_iter, len(train)),
            study_failures=False,
            strategy="definitions",
            sample_kinds=("definition", "we_call"),
            use_exam_paraphrase=False,
            seed=loop_seed,
            enable_visual_delivery=os.environ.get("ALLM_VISUAL_DELIVERY", "0") == "1",
        ),
    )

    if verbose:
        print("\n=== Kids KEL-steered held-out loop (M4) ===")
        print(f"  mission: {'on' if mission_enabled else 'off'}")
        if mission_enabled and identities:
            identity = next(iter(identities.values()))
            print(f"  identity: {resolved_identity}")
            print(f"  primary domains: {', '.join(identity.primary_domains[:6])}")
        print(f"  loop seed: {loop_seed}")
        print(f"  iterations (max): {loop_iterations}")
        if phases:
            print("\n=== Phased learning order (M37) ===")
            if books_only_mode():
                print("  mode: books-only (no workshop transcripts)")
            for phase in phases:
                print(f"  {phase.source}: {phase.iterations} iteration(s)")
            if disc_order := discovery_order_from_phase():
                print(f"  discovery order: {disc_order}")
        print("\n=== Hold-out curriculum gap (pre-loop) ===")
        print(format_holdout_gap_report(gap))

    first_score = last_score = 0.0
    peak_score = 0.0
    reports = loop.run_phased(phases) if phases else loop.run()
    for report in reports:
        row = report.students[0]
        first_score = first_score or row.score_before
        last_score = row.score_after
        peak_score = max(peak_score, row.score_after)
        if verbose:
            print(
                f"\n  iter {report.iteration}: "
                f"strategy={row.strategy} "
                f"{row.score_before:.2f} -> {row.score_after:.2f} "
                f"studied={row.samples_studied}"
            )

    kel_report = kel.evaluate()
    if researcher is not None:
        ecosystem = researcher.ecosystem_metrics(graph, state)
        kel.evaluate(ecosystem=ecosystem)
        if verbose:
            print("\n=== Researcher → KEL ecosystem ===")
            print(f"  missing_knowledge: {ecosystem.missing_knowledge:.2f}")
            print(f"  high_conflict_areas: {ecosystem.high_conflict_areas:.2f}")
            for finding in kel.diagnose():
                if finding.mode.startswith("research") or finding.mode == "high_conflict_discovery":
                    print(f"  KEL [{finding.mode}]: {finding.detail}")
    if verbose and loop._history is not None:
        history_records = loop._history.load_all()
        export_strategy_phase_gains(
            loop._history.path.with_name("strategy_phase_gains.json"),
            history_records,
        )
        print("\n=== Marginal learning gain by strategy ===")
        print(format_strategy_gain_report(compute_marginal_strategy_gains(history_records)))
        print(f"\nArtifacts: {run_dir}")

    visual_notes_delivered = 0
    if os.environ.get("ALLM_VISUAL_DELIVERY", "0") == "1":
        from allm.teacher.student_visual_delivery import count_visual_notes_delivered

        visual_notes_delivered = count_visual_notes_delivered(memory, student.student_id)

    if teacher_paused:
        from allm.teacher.teacher_kel_session import TeacherKelSessionStore

        TeacherKelSessionStore(store).mark_complete()

    store.close()
    history_path = run_dir / "learning_history.jsonl"
    return KidsKelRunResult(
        workdir=run_dir,
        first_score=first_score,
        last_score=last_score,
        iterations_completed=len(reports),
        peak_score=peak_score,
        mission_enabled=mission_enabled,
        identity_path=resolved_identity,
        loop_seed=loop_seed,
        history_path=history_path,
        kel_lg=kel_report.lg,
        kel_ks=getattr(loop, "_last_ks", None) or kel._last("ks"),
        workshop_packages=workshop_packages,
        book_packages=book_packages,
        aligned_concepts=aligned_concepts,
        book_figures=book_figures,
        student_visual_exports=student_visual_exports,
        visual_notes_delivered=visual_notes_delivered,
        teacher_approved_briefs=teacher_approved_briefs,
        teacher_rejected_briefs=teacher_rejected_briefs,
        teacher_paused=teacher_paused,
        multimodal_synced=multimodal_synced_count,
        learning_phase_order=phase_order_key,
    )
