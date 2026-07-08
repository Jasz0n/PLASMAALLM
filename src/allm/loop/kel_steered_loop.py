"""KEL-steered variant of the continuous learning loop."""

from __future__ import annotations

import os
from pathlib import Path

from allm.core.logging import get_logger
from allm.evaluation.generalization import HoldoutGapReport
from allm.kel.knowledge_stability import (
    cross_topic_coherence,
    debate_consistency_ks,
    knowledge_stability,
    ks_from_forgetting,
    mastery_stability_ks,
    merge_ks,
    retrieval_ks_from_state,
)
from allm.kel.layer import KnowledgeEvaluationLayer
from allm.loop.history import IterationHistoryWriter, LearningIterationRecord, LearningRunManifest
from allm.loop.kel_steering import (
    KelSteeringConfig,
    KelSteeringPolicy,
    apply_steering,
)
from allm.loop.learning_loop import IterationReport, LearningLoop, LoopConfig
from allm.loop.phased_learning import LearningPhase
from allm.loop.retention_gates import (
    HeldoutRetentionTracker,
    build_retention_context,
    reset_strategy_for_new_phase,
)

logger = get_logger("loop.kel")


class KelSteeredLearningLoop(LearningLoop):
    """Learning loop that calls KEL each iteration and adapts behaviour."""

    def __init__(
        self,
        *,
        kel: KnowledgeEvaluationLayer,
        steering: KelSteeringConfig | None = None,
        history_path: Path | str | None = None,
        holdout_gap: HoldoutGapReport | None = None,
        run_manifest: LearningRunManifest | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._kel = kel
        self._policy = KelSteeringPolicy(steering)
        self._history = (
            IterationHistoryWriter(history_path) if history_path is not None else None
        )
        self._holdout_gap = holdout_gap
        self._base_samples = self._config.samples_per_iteration
        self._retention_tracker = HeldoutRetentionTracker()
        self._last_ks: float | None = None
        if self._history is not None and run_manifest is not None:
            self._history.write_manifest(run_manifest)

    def run(self) -> list[IterationReport]:
        return self.run_phased(None)

    def run_phased(self, phases: tuple[LearningPhase, ...] | None) -> list[IterationReport]:
        """Run the loop, optionally in book/workshop phases."""
        reports: list[IterationReport] = []
        active = self._config
        segments: list[tuple[str | None, int]] = (
            [(phase.source, phase.iterations) for phase in phases]
            if phases
            else [(None, self._config.iterations)]
        )
        iteration = 0
        segment_active = active
        for source, segment_iters in segments:
            if source is not None:
                segment_active = reset_strategy_for_new_phase(segment_active)
                segment_active = segment_active.model_copy(
                    update={
                        "learning_source": source,
                        "samples_per_iteration": self._base_samples,
                    }
                )
                if iteration > 0:
                    logger.info(
                        "KEL phase transition -> %s: reset strategy=definitions samples=%d",
                        source,
                        self._base_samples,
                    )
            for _ in range(segment_iters):
                iteration += 1
                retention = build_retention_context(
                    reports,
                    self._retention_tracker,
                    current_phase=source,
                    max_drop_from_peak=self._policy._config.retention_max_drop_from_peak,
                    require_stable=self._policy._config.require_retention_stable,
                    ks=self._last_ks,
                    ks_threshold=float(os.environ.get("ALLM_KS_ADVANCE_THRESHOLD", "0.70")),
                )
                if not retention.retention_stable:
                    logger.info(
                        "KEL retention gate: %s",
                        retention.block_reason or "unstable",
                    )
                eval_topics = tuple(
                    dict.fromkeys(
                        (
                            *(reports[-1].students[0].goals if reports and reports[-1].students else ()),
                            *(s.specialty for s in self._students),
                        )
                    )
                )
                from allm.evaluator.context import build_evaluation_input
                from allm.evaluator.independent import IndependentEvaluator
                from allm.kel.objectives import compromise_decision, multi_objective_kel_enabled

                compromise = None
                if multi_objective_kel_enabled():
                    eval_input = build_evaluation_input(
                        state=self._teacher.state,
                        student_id=self._students[0].student_id,
                        topics=eval_topics,
                        reports=reports,
                        kel=self._kel,
                        retention=retention,
                        kel_ks=self._last_ks,
                    )
                    snapshot = IndependentEvaluator().evaluate(eval_input)
                    compromise = compromise_decision(snapshot)
                decision = self._policy.decide(
                    iteration,
                    reports,
                    self._kel,
                    segment_active,
                    retention=retention,
                    compromise=compromise,
                )
                for finding in decision.findings:
                    logger.info("KEL finding [%s]: %s", finding.mode, finding.detail)
                if decision.halt:
                    logger.warning(
                        "KEL steering halted before iteration %d: %s",
                        iteration,
                        decision.reason,
                    )
                    if self._run is not None:
                        self._run.finish()
                    return reports
                previous_strategy = segment_active.strategy
                segment_active = apply_steering(segment_active, decision)
                if segment_active.strategy != previous_strategy:
                    logger.info(
                        "KEL strategy transition: %s -> %s",
                        previous_strategy,
                        segment_active.strategy,
                    )
                scores = self._policy._recent_scores(reports)
                peak = max(scores) if scores else 0.0
                rolling = sum(scores) / len(scores) if scores else 0.0
                phase_label = source or "all"
                logger.info(
                    "KEL steering iter %d phase=%s: next=%s peak=%.2f rolling=%.2f "
                    "questions=%d samples=%d paraphrase=%s",
                    iteration,
                    phase_label,
                    segment_active.strategy,
                    peak,
                    rolling,
                    segment_active.questions_per_exam,
                    segment_active.samples_per_iteration,
                    segment_active.use_exam_paraphrase,
                )
                self._global_ks = self._last_ks
                report = self._iteration(iteration, config=segment_active)
                reports.append(report)
                if report.students:
                    self._retention_tracker.record(
                        report.students[0].score_after,
                        phase=source,
                    )
                student_row = report.students[0] if report.students else None
                if student_row is not None:
                    topics = tuple(dict.fromkeys((*student_row.goals, student_row.student_id)))
                    student = next(
                        (row for row in self._students if row.student_id == student_row.student_id),
                        None,
                    )
                    specialty_topics = (student.specialty,) if student is not None else ()
                    topics = tuple(dict.fromkeys((*topics, *specialty_topics)))
                    confidence_ks = knowledge_stability(
                        self._teacher.state,
                        student_row.student_id,
                        topics,
                    )
                    forgetting_ks = ks_from_forgetting(report.forgetting)
                    retrieval_ks = retrieval_ks_from_state(
                        self._teacher.state,
                        student_row.student_id,
                        topics,
                    )
                    mastery_ks = mastery_stability_ks(
                        self._teacher.state,
                        student_row.student_id,
                        topics,
                    )
                    cross_topic_ks = cross_topic_coherence(
                        self._teacher.state,
                        student_row.student_id,
                        topics,
                    )
                    debate_ks = debate_consistency_ks(report.debate_disagreement)
                    self._last_ks = merge_ks(
                        confidence_ks,
                        forgetting_ks,
                        retrieval_ks,
                        mastery_ks,
                        cross_topic_ks,
                        debate_ks,
                    )
                    self._kel.record_stability(self._last_ks)
                    if self._last_ks is not None:
                        logger.info(
                            "KEL knowledge stability iter %d: KS=%.2f",
                            iteration,
                            self._last_ks,
                        )
                researcher = getattr(self, "_researcher", None)
                if researcher is not None and hasattr(researcher, "set_context"):
                    researcher.set_context(
                        graph=self._graph,
                        state=self._teacher.state,
                        student_ids=tuple(s.student_id for s in self._students),
                        kel_findings=decision.findings,
                    )
                kel_report = self._kel.evaluate(
                    ecosystem=self._researcher_ecosystem(),
                )
                self._record_history(
                    report,
                    segment_active,
                    kel_report,
                    decision.findings,
                    strategy_previous=previous_strategy,
                )
                if student_row is not None:
                    from allm.kel.research_requests import (
                        build_kel_research_requests,
                        kel_research_requests_enabled,
                    )

                    if kel_research_requests_enabled() and researcher is not None:
                        retention_after = build_retention_context(
                            reports,
                            self._retention_tracker,
                            current_phase=source,
                            max_drop_from_peak=self._policy._config.retention_max_drop_from_peak,
                            require_stable=self._policy._config.require_retention_stable,
                            ks=self._last_ks,
                            ks_threshold=float(
                                os.environ.get("ALLM_KS_ADVANCE_THRESHOLD", "0.70")
                            ),
                        )
                        compromise_mode = decision.compromise_mode
                        if multi_objective_kel_enabled():
                            eval_input = build_evaluation_input(
                                state=self._teacher.state,
                                student_id=student_row.student_id,
                                topics=topics,
                                reports=reports,
                                kel=self._kel,
                                retention=retention_after,
                                kel_ks=self._last_ks,
                            )
                            compromise_mode = compromise_decision(
                                IndependentEvaluator().evaluate(eval_input)
                            ).mode
                        requests = build_kel_research_requests(
                            findings=decision.findings,
                            compromise_mode=compromise_mode,
                            retention=retention_after,
                            reports=reports,
                            student_id=student_row.student_id,
                            topics=topics,
                            strategy=segment_active.strategy,
                            kel_ks=self._last_ks,
                        )
                        if hasattr(researcher, "submit_kel_research_requests") and requests:
                            from allm.researcher.curriculum_diagnostics import DiagnosticContext

                            history_records = (
                                self._history.load_all() if self._history is not None else []
                            )
                            latest = history_records[-1] if history_records else None
                            conflicts = None
                            last_report = getattr(researcher, "_last_report", None)
                            if last_report is not None:
                                conflicts = last_report.conflicts_detected
                            diagnostic_context = DiagnosticContext(
                                failure_prompts=latest.failure_prompts if latest else (),
                                strategy=segment_active.strategy,
                                kel_ks=self._last_ks,
                                conflict_count=conflicts,
                                history=tuple(history_records),
                            )
                            count = researcher.submit_kel_research_requests(
                                requests,
                                diagnostic_context=diagnostic_context,
                            )
                            if count:
                                logger.info(
                                    "KEL submitted %d research request(s) to Researcher",
                                    count,
                                )
                                for request in requests[:3]:
                                    logger.info(
                                        "  [%s] %s: %s",
                                        request.trigger,
                                        request.topic,
                                        request.task[:80],
                                    )
        if self._run is not None:
            self._run.finish()
        return reports

    def _researcher_ecosystem(self):
        """Researcher metrics for KEL when a ResearcherLayer is attached."""
        researcher = getattr(self, "_researcher", None)
        if researcher is None or self._graph is None:
            return None
        return researcher.ecosystem_metrics(self._graph, self._teacher.state)

    def _record_history(
        self,
        report: IterationReport,
        active: LoopConfig,
        kel_report,
        findings,
        *,
        strategy_previous: str | None = None,
    ) -> None:
        if self._history is None or not report.students:
            return
        row = report.students[0]
        failure_prompts: list[str] = []
        for exam in self._teacher.state.exam_results(row.student_id)[-2:]:
            for result in exam.results:
                if not result.correct:
                    failure_prompts.append(result.question.prompt)
        record = LearningIterationRecord(
            iteration=report.iteration,
            strategy=active.strategy,  # type: ignore[arg-type]
            sample_kinds=active.sample_kinds,
            sample_ids=row.sample_ids,
            student_id=row.student_id,
            score_before=row.score_before,
            score_after=row.score_after,
            goals=row.goals,
            samples_studied=row.samples_studied,
            questions_per_exam=active.questions_per_exam,
            use_exam_paraphrase=active.use_exam_paraphrase,
            kel_lg=kel_report.lg,
            kel_ghs=kel_report.ghs,
            kel_findings=tuple(f.mode for f in findings),
            failure_prompts=tuple(failure_prompts[:8]),
            holdout_count=self._holdout_gap.holdout_count if self._holdout_gap else None,
            holdout_answers_in_train=(
                self._holdout_gap.answers_in_train if self._holdout_gap else None
            ),
            holdout_novel_lexical=(
                self._holdout_gap.novel_lexical if self._holdout_gap else None
            ),
            strategy_previous=strategy_previous,  # type: ignore[arg-type]
            strategy_changed=(
                strategy_previous is not None and strategy_previous != active.strategy
            ),
            metadata={"kel_ks": self._last_ks},
        )
        self._history.append(record)
        researcher = getattr(self, "_researcher", None)
        if researcher is not None and hasattr(researcher, "_store"):
            import json

            researcher._store.put(
                "learning_history",
                str(record.iteration),
                json.loads(record.model_dump_json()),
                reason="learning iteration",
            )
