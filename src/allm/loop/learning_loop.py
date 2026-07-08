"""The continuous learning loop — Plan.md's while(true), made explicit.

    measure -> plan -> collect -> learn -> debate -> test
            -> compress -> update memory -> repeat

Design decisions
----------------
- The loop is the *composition root*: it owns no logic of its own, only
  the order of calls into injected components. Every step below maps
  1:1 to a Plan.md stage and to one component, so any stage can be
  swapped or disabled (graph/compression and debate are optional).
- Each iteration measures **before** learning and tests **after**
  learning with a fresh exam, so the per-iteration delta is a real
  learning signal, not memorisation of the measure exam.
- Everything observable is recorded as it happens: exams into teacher
  state, mistakes into the failure log, outcomes into episodic memory,
  scalars into the experiment tracker. The loop never keeps private
  state a crash would lose.
"""

from __future__ import annotations

import os
import random

from pydantic import BaseModel, ConfigDict, Field

from allm.collector.pool import SamplePool
from allm.data.base import Sample
from allm.core.logging import get_logger
from allm.compression.engine import CompressionEngine
from allm.debate.engine import DebateEngine
from allm.knowledge.graph import KnowledgeGraph
from allm.memory.episodic import EpisodicMemory
from allm.planner.base import Planner
from allm.planner.signals import build_signals
from allm.students.identity import StudentIdentity, domain_fit
from allm.students.failures import FailureLog
from allm.students.model_student import ModelStudent
from allm.teacher.consultation import consultation_samples, mediated_consultation_samples
from allm.teacher.teacher import Teacher
from allm.tracking.base import Run
from allm.trainer.base import Trainer
from allm.trainer.forgetting import ForgettingReport, ForgettingWatchdog

logger = get_logger("loop")


class LoopConfig(BaseModel):
    """Loop tunables."""

    model_config = ConfigDict(frozen=True)

    iterations: int = Field(default=3, ge=1)
    questions_per_exam: int = Field(default=6, ge=1)
    max_goals: int = Field(default=3, ge=1)
    samples_per_iteration: int = Field(default=16, ge=1)
    study_failures: bool = Field(default=True)
    seed: int = 0
    strategy: str = Field(default="definitions")
    # Empty means "no kind filter": plain samples without sample_kind
    # metadata stay collectable. Corpus-specific kinds are opt-in via
    # strategy profiles (loop/strategy.py) or explicit configs.
    sample_kinds: tuple[str, ...] = ()
    use_exam_paraphrase: bool = False
    enable_debate_evidence: bool = False
    enable_consult_show_me: bool = False
    enable_visual_delivery: bool = False
    learning_source: str = "all"


class StudentIteration(BaseModel):
    """One student's numbers for one iteration."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    score_before: float
    score_after: float
    goals: tuple[str, ...]
    samples_studied: int
    strategy: str = "definitions"
    sample_ids: tuple[str, ...] = ()


class IterationReport(BaseModel):
    """Everything one loop iteration did."""

    model_config = ConfigDict(frozen=True)

    iteration: int
    students: tuple[StudentIteration, ...]
    debate_disagreement: float | None
    debate_evidence: object | None = None
    compression_applied: int
    compression_retracted: int
    forgetting: tuple[ForgettingReport, ...] = ()


class LearningLoop:
    """Runs the measure->...->repeat cycle over injected components."""

    def __init__(
        self,
        *,
        teacher: Teacher,
        students: list[ModelStudent],
        planner: Planner,
        trainer: Trainer,
        pool: SamplePool,
        memory: EpisodicMemory,
        failure_log: FailureLog,
        graph: KnowledgeGraph | None = None,
        compression: CompressionEngine | None = None,
        debate: DebateEngine | None = None,
        forgetting: ForgettingWatchdog | None = None,
        run: Run | None = None,
        config: LoopConfig | None = None,
        identities: dict[str, StudentIdentity] | None = None,
        enable_peer_consultation: bool = False,
        enable_mediated_consultation: bool = False,
        researcher: object | None = None,
    ) -> None:
        if not students:
            raise ValueError("the loop needs at least one student")
        self._teacher = teacher
        self._students = students
        self._planner = planner
        self._trainer = trainer
        self._pool = pool
        self._memory = memory
        self._failures = failure_log
        self._graph = graph
        self._compression = compression
        self._debate = debate
        self._forgetting = forgetting
        self._run = run
        self._config = config or LoopConfig()
        self._identities = identities or {}
        self._enable_peer_consultation = enable_peer_consultation
        self._enable_mediated_consultation = enable_mediated_consultation
        self._researcher = researcher
        self._global_ks: float | None = None

    def run(self) -> list[IterationReport]:
        reports = [self._iteration(i) for i in range(1, self._config.iterations + 1)]
        if self._run is not None:
            self._run.finish()
        return reports

    def _iteration(self, iteration: int, *, config: LoopConfig | None = None) -> IterationReport:
        cfg = config or self._config
        self._teacher.set_exam_paraphrase(cfg.use_exam_paraphrase)
        logger.info("=== iteration %d (strategy=%s) ===", iteration, cfg.strategy)
        seed = cfg.seed * 10_000 + iteration
        base_catalog = self._graph.to_catalog() if self._graph is not None else {}
        if self._researcher is not None and hasattr(self._researcher, "set_catalog_topics"):
            self._researcher.set_catalog_topics(set(base_catalog.keys()))

        student_reports = []
        forgetting_reports = []
        for student in self._students:
            catalog = dict(base_catalog)
            if self._researcher is not None:
                from allm.planner.researcher_signals import merge_research_recommendations
                from allm.loop.maintenance_curriculum import maintenance_topics_from_recommendations

                recs = self._researcher.active_recommendations()
                if os.environ.get("ALLM_RESEARCHER_TARGETING", "0") == "1":
                    recs = self._researcher.active_recommendations(student_id=student.student_id)
                catalog = merge_research_recommendations(catalog, recs)
                maintenance_topic_list = maintenance_topics_from_recommendations(recs)
            else:
                recs = []
                maintenance_topic_list = []

            # 1. Measure.
            exam = self._teacher.create_exam(
                num_questions=cfg.questions_per_exam, seed=seed
            )
            before = self._teacher.evaluate(student, exam)
            self._memory.remember_exam(before)
            for failed in before.failures():
                self._failures.record(student.student_id, failed)

            # 2. Plan (includes forgetting risk when KS planner is enabled).
            signals = build_signals(
                self._teacher.state,
                student.student_id,
                dict(catalog),
                identity=self._identities.get(student.student_id),
                mission_seed=cfg.seed * 10_000 + iteration,
                maintenance_topics=maintenance_topic_list,
                global_ks=self._global_ks,
                graph=self._graph,
            )
            roadmap = self._planner.plan(student.student_id, signals)
            goals = roadmap.to_goals(max_goals=cfg.max_goals)
            self._teacher.state.record_goals(student.student_id, goals)
            goal_topics = [g.topic for g in goals]
            from allm.planner.forgetting_risk import review_topics_from_roadmap
            from allm.planner.maintenance_budget import optimized_review_topics

            if os.environ.get("ALLM_MAINTENANCE_OPTIMIZER", "1") == "1":
                planner_review_topics = optimized_review_topics(
                    roadmap.items,
                    state=self._teacher.state,
                    student_id=student.student_id,
                    graph=self._graph,
                )
            else:
                planner_review_topics = review_topics_from_roadmap(roadmap.items)
            # 3. Collect: reserve review budget from full iteration size (M41).
            book_samples: list[Sample] = []
            if cfg.learning_source in {"book", "all"} and self._researcher is not None:
                from allm.teacher.source_training import samples_from_book_packages

                packages = tuple(self._researcher.stored_packages())
                book_samples = samples_from_book_packages(
                    packages,
                    topic=student.specialty,
                    limit=cfg.samples_per_iteration,
                )
            split = None
            if os.environ.get("ALLM_MAINTENANCE_CURRICULUM", "1") == "1":
                from allm.loop.maintenance_curriculum import (
                    collect_curriculum_mix,
                    maintenance_split_from_env,
                )

                split = maintenance_split_from_env(self._global_ks)
            collected: list[Sample] = []
            maintenance_counts: dict[str, int] | None = None
            if split is not None and cfg.learning_source in {"book", "workshop", "all"}:
                primary = book_samples if cfg.learning_source in {"book", "all"} else []
                collected, maintenance_counts = collect_curriculum_mix(
                    pool=self._pool,
                    failures=self._failures,
                    state=self._teacher.state,
                    student_id=student.student_id,
                    goal_topics=goal_topics,
                    maintenance_topics=maintenance_topic_list,
                    cfg=cfg,
                    split=split,
                    primary_samples=primary,
                    planner_review_topics=planner_review_topics,
                )
                phase_label = cfg.learning_source
                logger.info(
                    "%s curriculum (KS=%s): primary=%d new=%d review=%d difficult=%d split=%.0f/%.0f/%.0f",
                    phase_label,
                    f"{self._global_ks:.2f}" if self._global_ks is not None else "n/a",
                    maintenance_counts.get("primary", 0),
                    maintenance_counts["new"],
                    maintenance_counts["review"],
                    maintenance_counts["difficult"],
                    split.new_fraction * 100,
                    split.review_fraction * 100,
                    split.difficult_fraction * 100,
                )
            elif cfg.learning_source in {"workshop", "all"}:
                collected.extend(
                    self._pool.collect(
                        topics=goal_topics or None,
                        limit=cfg.samples_per_iteration,
                        kinds=cfg.sample_kinds or None,
                    )
                )
                if book_samples and cfg.learning_source == "all":
                    collected = (book_samples + collected)[: cfg.samples_per_iteration]
            if (
                cfg.learning_source == "workshop"
                and os.environ.get("ALLM_WORKSHOP_DELTA", "0") == "1"
                and self._researcher is not None
            ):
                from allm.teacher.source_training import filter_workshop_delta_samples

                collected = filter_workshop_delta_samples(
                    collected,
                    tuple(self._researcher.stored_packages()),
                )
            identity = self._identities.get(student.student_id)
            if identity is not None:
                mission_seed = cfg.seed * 10_000 + iteration
                collected = self._filter_by_mission(collected, identity, mission_seed)
            if self._enable_mediated_consultation and identity is not None:
                mission_seed = cfg.seed * 10_000 + iteration
                experts = {row.student_id: row for row in self._students}
                consult_broker = None
                if cfg.enable_consult_show_me and self._researcher is not None:
                    consult_broker = self._researcher.evidence_broker()
                mediated_samples, mediated_results = mediated_consultation_samples(
                    self._teacher.state,
                    self._teacher.grader,
                    self._pool,
                    student.student_id,
                    student,
                    identity,
                    before,
                    experts,
                    mission_seed=mission_seed,
                    evidence_broker=consult_broker,
                    show_me_on_reject=cfg.enable_consult_show_me,
                )
                for result in mediated_results:
                    if result.expert_id is not None:
                        logger.info(
                            "mediated consult: %s asked %s about %r approved=%s",
                            result.asker_id,
                            result.expert_id,
                            result.topic,
                            result.approved,
                        )
                        self._memory.remember(
                            student.student_id,
                            "observation",
                            f"mediated consult {result.expert_id} on {result.topic}: "
                            f"{'approved' if result.approved else 'rejected'}",
                            topic=result.topic,
                            detail={
                                "expert_id": result.expert_id,
                                "approved": result.approved,
                                "reason": result.reason,
                            },
                        )
                        if result.show_me_requested and result.evidence is not None:
                            evidence = result.evidence
                            if getattr(evidence, "found", False):
                                logger.info(
                                    "consult show me: %s query=%r hits=%d",
                                    result.asker_id,
                                    getattr(evidence, "query", ""),
                                    getattr(evidence, "hit_count", 0),
                                )
                                self._memory.remember(
                                    student.student_id,
                                    "observation",
                                    f"show me evidence: {getattr(evidence, 'summary', '')}",
                                    topic=result.topic,
                                    detail={
                                        "query": getattr(evidence, "query", ""),
                                        "confidence": getattr(evidence, "confidence", 0.0),
                                        "hit_count": getattr(evidence, "hit_count", 0),
                                    },
                                )
                collected.extend(mediated_samples)
            elif self._enable_peer_consultation and identity is not None:
                mission_seed = cfg.seed * 10_000 + iteration
                peer_samples, consultations = consultation_samples(
                    self._teacher.state,
                    self._pool,
                    student.student_id,
                    identity,
                    before,
                    mission_seed=mission_seed,
                )
                for request in consultations:
                    if request.expert_id is not None:
                        logger.info(
                            "peer consult: %s asked %s about %r (%s)",
                            request.asker_id,
                            request.expert_id,
                            request.topic,
                            request.reason,
                        )
                        self._memory.remember(
                            student.student_id,
                            "observation",
                            f"consulted {request.expert_id} on {request.topic}",
                            topic=request.topic,
                            detail={"expert_id": request.expert_id, "reason": request.reason},
                        )
                collected.extend(peer_samples)
            if cfg.study_failures and maintenance_counts is None:
                collected += self._failures.training_samples(student.student_id)
            if len(collected) > cfg.samples_per_iteration:
                collected = collected[: cfg.samples_per_iteration]

            if cfg.enable_visual_delivery and self._researcher is not None:
                from allm.teacher.source_training import deliver_visuals_for_source

                source = cfg.learning_source if cfg.learning_source in {"book", "workshop"} else "all"
                delivered = deliver_visuals_for_source(self._researcher, student, source)
                if delivered:
                    logger.info(
                        "%s received %d visual study note(s) from Teacher-approved packages",
                        student.student_id,
                        delivered,
                    )
                    self._memory.remember(
                        student.student_id,
                        "observation",
                        f"Teacher delivered {delivered} visual study note(s)",
                        topic=student.specialty,
                        detail={"visual_notes": delivered},
                    )

            # 4. Learn.
            mastered_before = (
                self._forgetting.mastered_topics(self._teacher.state, student.student_id)
                if self._forgetting is not None
                else {}
            )
            training = self._trainer.train(student, collected)

            if self._forgetting is not None and mastered_before:
                report = self._forgetting.check(
                    student, mastered_before, seed=seed + 700_000
                )
                forgetting_reports.append(report)
                if report.regressions:
                    self._memory.remember(
                        student.student_id,
                        "observation",
                        f"forgetting detected: {report.regressions}",
                        detail={"regressions": report.regressions},
                    )

            # 6. Test (fresh exam — measuring learning, not memorisation
            # of the measure exam).
            test = self._teacher.create_exam(
                num_questions=cfg.questions_per_exam, seed=seed + 500_000
            )
            after = self._teacher.evaluate(student, test)
            self._memory.remember_exam(after)
            for failed in after.failures():
                self._failures.record(student.student_id, failed)

            self._log_metrics(
                student.student_id, iteration, before.score, after.score
            )
            student_reports.append(
                StudentIteration(
                    student_id=student.student_id,
                    score_before=before.score,
                    score_after=after.score,
                    goals=tuple(goal_topics),
                    samples_studied=training.samples_used,
                    strategy=cfg.strategy,
                    sample_ids=tuple(sample.id for sample in collected),
                )
            )

        # 5. Debate (one contested question per iteration, all students).
        disagreement, debate_evidence = self._run_debate(seed) if len(self._students) > 1 else (None, None)

        # 7. Compress.
        applied = retracted = 0
        if self._compression is not None:
            for outcome in self._compression.compress():
                applied += outcome.applied and not outcome.retracted
                retracted += outcome.retracted

        # 8. Update memory with the iteration summary.
        self._memory.remember(
            "loop",
            "observation",
            f"iteration {iteration}: "
            + ", ".join(
                f"{r.student_id} {r.score_before:.2f}->{r.score_after:.2f}"
                for r in student_reports
            ),
            detail={"iteration": iteration},
        )
        return IterationReport(
            iteration=iteration,
            students=tuple(student_reports),
            debate_disagreement=disagreement,
            debate_evidence=debate_evidence,
            compression_applied=applied,
            compression_retracted=retracted,
            forgetting=tuple(forgetting_reports),
        )

    def _run_debate(self, seed: int):
        if self._debate is None:
            return None, None
        exam = self._teacher.create_exam(num_questions=1, seed=seed + 900_000)
        question = random.Random(seed).choice(exam.questions)
        result = self._debate.debate(question, self._students)
        debate_evidence = None
        if result.unresolved:
            self._pool.ingest([result.to_learning_sample()])
            if self._config.enable_debate_evidence:
                debate_evidence = self._resolve_debate_evidence(result)
            self._memory.remember(
                "loop",
                "observation",
                f"unresolved debate on {question.prompt!r} "
                f"(disagreement {result.disagreement:.2f}) -> learning task",
                topic=question.topic,
                detail={"disagreement": result.disagreement},
            )
            if debate_evidence is not None and debate_evidence.found:
                self._memory.remember(
                    "loop",
                    "observation",
                    f"debate evidence: {debate_evidence.summary}",
                    topic=question.topic,
                    detail={
                        "query": debate_evidence.query,
                        "confidence": debate_evidence.confidence,
                        "hit_count": debate_evidence.hit_count,
                    },
                )
                logger.info(
                    "debate evidence: query=%r hits=%d conf=%.2f",
                    debate_evidence.query,
                    debate_evidence.hit_count,
                    debate_evidence.confidence,
                )
        return result.disagreement, debate_evidence

    def _resolve_debate_evidence(self, result):
        researcher = self._researcher
        if researcher is None or not hasattr(researcher, "evidence_broker"):
            return None
        from allm.loop.debate_evidence import resolve_loop_debate_evidence

        broker = researcher.evidence_broker()
        return resolve_loop_debate_evidence(broker, result)

    def _log_metrics(
        self, student_id: str, iteration: int, before: float, after: float
    ) -> None:
        if self._run is None:
            return
        self._run.log_metric(f"{student_id}/score_before", before, step=iteration)
        self._run.log_metric(f"{student_id}/score_after", after, step=iteration)

    @staticmethod
    def _filter_by_mission(
        samples: list[Sample],
        identity: StudentIdentity,
        mission_seed: int,
    ) -> list[Sample]:
        """Drop samples outside the student's mission (specialist focus)."""
        kept: list[Sample] = []
        for sample in samples:
            topic = str(sample.metadata.get("topic", "general"))
            fit, _reason = domain_fit(topic, identity, seed=mission_seed)
            if fit > 0.0:
                kept.append(sample)
        return kept
