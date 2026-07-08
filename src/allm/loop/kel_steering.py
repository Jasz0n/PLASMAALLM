"""KEL-driven adjustments for the continuous learning loop (KEL.md section 10)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.kel.layer import KnowledgeEvaluationLayer
from allm.kel.types import Finding
from allm.loop.capability_progression import capability_allows_advance, capability_progression_enabled
from allm.loop.learning_loop import IterationReport, LoopConfig
from allm.loop.retention_gates import RetentionContext
from allm.loop.strategy import LearningStrategy, advance_strategy, profile_for

if TYPE_CHECKING:
    from allm.kel.objectives import CompromiseDecision

logger = get_logger("loop.kel")


class KelSteeringConfig(BaseModel):
    """Tunables for KEL-guided loop behaviour."""

    model_config = ConfigDict(frozen=True)

    lg_window: int = Field(default=3, ge=2)
    mastery_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    strategy_advance_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    strategy_advance_window: int = Field(default=3, ge=1)
    stagnation_iterations: int = Field(default=3, ge=2)
    max_questions: int = Field(default=16, ge=1)
    max_samples: int = Field(default=128, ge=1)
    sample_boost: float = Field(default=1.5, gt=1.0)
    question_boost: int = Field(default=2, ge=1)
    min_iterations_before_halt: int = Field(default=6, ge=1)
    min_lg_history_for_halt: int = Field(default=5, ge=2)
    halt_on_static_illusion: bool = True
    require_retention_stable: bool = True
    retention_max_drop_from_peak: float = Field(default=0.15, ge=0.0, le=1.0)
    block_advance_on_forgetting: bool = True
    cap_samples_when_unstable: bool = True


class KelSteeringDecision(BaseModel):
    """Per-iteration loop adjustments derived from KEL measurements."""

    model_config = ConfigDict(frozen=True)

    halt: bool = False
    reason: str = ""
    strategy: LearningStrategy | None = None
    questions_per_exam: int | None = None
    samples_per_iteration: int | None = None
    max_goals: int | None = None
    findings: tuple[Finding, ...] = ()
    compromise_mode: str | None = None
    compromise_score: float | None = None
    compromise_reason: str = ""


class KelSteeringPolicy:
    """Translate KEL trends and failure modes into loop controls."""

    def __init__(self, config: KelSteeringConfig | None = None) -> None:
        self._config = config or KelSteeringConfig()

    def decide(
        self,
        iteration: int,
        reports: list[IterationReport],
        kel: KnowledgeEvaluationLayer,
        active: LoopConfig,
        retention: RetentionContext | None = None,
        compromise: "CompromiseDecision | None" = None,
    ) -> KelSteeringDecision:
        """Return steering adjustments before ``iteration`` runs."""
        from allm.kel.objectives import multi_objective_kel_enabled

        findings = tuple(kel.diagnose())
        decision = KelSteeringDecision(findings=findings)
        if compromise is not None and multi_objective_kel_enabled():
            decision = decision.model_copy(
                update={
                    "compromise_mode": compromise.mode,
                    "compromise_score": compromise.compromise_score,
                    "compromise_reason": compromise.reason,
                }
            )
            logger.info(
                "KEL multi-objective: mode=%s score=%.2f (%s)",
                compromise.mode,
                compromise.compromise_score,
                compromise.reason,
            )
        current = LearningStrategy(active.strategy)

        for finding in findings:
            if finding.mode == "static_illusion" and self._should_halt_static_illusion(
                iteration, kel, reports, retention
            ):
                return decision.model_copy(update={"halt": True, "reason": finding.detail})
            if finding.mode == "unstable_mastery" and self._should_repair_instead_of_halt(
                retention
            ):
                logger.info(
                    "KEL unstable mastery: continuing with maintenance curriculum (%s)",
                    finding.detail,
                )

        updates: dict = {}
        strategy_update = self._pick_strategy(
            current, reports, findings, retention, compromise=compromise
        )
        if strategy_update is not None and strategy_update != current:
            updates["strategy"] = strategy_update

        from allm.kel.objectives import multi_objective_kel_enabled

        if compromise is not None and multi_objective_kel_enabled():
            if compromise.mode in {"maintain", "repair"}:
                if self._config.cap_samples_when_unstable and "samples_per_iteration" not in updates:
                    factor = 0.75 if compromise.mode == "repair" else 0.85
                    updates["samples_per_iteration"] = max(
                        int(active.samples_per_iteration * factor),
                        16,
                    )

        if retention is not None and not retention.retention_stable:
            if self._config.cap_samples_when_unstable and "samples_per_iteration" not in updates:
                if reports and reports[-1].students:
                    last = reports[-1].students[0].score_after
                    if (
                        retention.heldout_peak > 0
                        and last < retention.heldout_peak - self._config.retention_max_drop_from_peak
                    ):
                        updates["samples_per_iteration"] = max(
                            int(active.samples_per_iteration * 0.75),
                            16,
                        )

        lg_trend = kel.trend("lg", window=self._config.lg_window)
        if (
            lg_trend is not None
            and lg_trend < 0
            and (retention is None or retention.retention_stable)
        ):
            updates["samples_per_iteration"] = min(
                int(active.samples_per_iteration * self._config.sample_boost),
                self._config.max_samples,
            )

        if reports and reports[-1].students:
            last_score = reports[-1].students[0].score_after
            if last_score >= self._config.mastery_threshold:
                updates["questions_per_exam"] = min(
                    active.questions_per_exam + self._config.question_boost,
                    self._config.max_questions,
                )

        stagnation = self._exam_stagnation(reports)
        if (
            stagnation
            and "samples_per_iteration" not in updates
            and (retention is None or retention.retention_stable)
        ):
            updates["samples_per_iteration"] = min(
                int(active.samples_per_iteration * self._config.sample_boost),
                self._config.max_samples,
            )

        for finding in findings:
            if finding.mode in {"dead_knowledge_growth", "conflict_accumulation"}:
                if finding.mode == "conflict_accumulation":
                    updates["strategy"] = LearningStrategy.RESEARCH
                if "samples_per_iteration" not in updates:
                    updates["samples_per_iteration"] = min(
                        int(active.samples_per_iteration * self._config.sample_boost),
                        self._config.max_samples,
                    )
                if finding.mode == "conflict_accumulation":
                    updates["max_goals"] = min(active.max_goals + 1, 6)
                break

        if not updates:
            return decision
        return decision.model_copy(update=updates)

    def _pick_strategy(
        self,
        current: LearningStrategy,
        reports: list[IterationReport],
        findings: tuple[Finding, ...],
        retention: RetentionContext | None = None,
        compromise: "CompromiseDecision | None" = None,
    ) -> LearningStrategy | None:
        from allm.kel.objectives import multi_objective_kel_enabled

        if compromise is not None and multi_objective_kel_enabled():
            if compromise.mode in {"maintain", "repair"}:
                from allm.kel.research_requests import _strategy_stagnation, strategy_stagnation_iters

                if (
                    os.environ.get("ALLM_KEL_STRATEGY_DIVERSITY", "1") == "1"
                    and current == LearningStrategy.DEFINITIONS
                    and _strategy_stagnation(
                        reports,
                        current.value,
                        min_iters=strategy_stagnation_iters(),
                    )
                ):
                    logger.info(
                        "KEL strategy diversity: advancing definitions -> relations "
                        "after %d stagnant iterations",
                        strategy_stagnation_iters(),
                    )
                    return LearningStrategy.RELATIONS
                return None
        for finding in findings:
            if finding.mode == "conflict_accumulation":
                if retention is not None and not retention.retention_stable:
                    return None
                return LearningStrategy.RESEARCH

        if not reports or not reports[-1].students:
            return LearningStrategy.DEFINITIONS if current != LearningStrategy.DEFINITIONS else None

        if retention is not None and not self._retention_allows_advance(retention):
            return None

        if capability_progression_enabled():
            allowed, _ = capability_allows_advance(
                current,
                reports,
                ks=None if retention is None else retention.ks,
            )
            if allowed:
                return advance_strategy(current)
            return None

        scores = self._recent_scores(reports)
        if scores:
            peak = max(scores)
            rolling = sum(scores) / len(scores)
            threshold = self._config.strategy_advance_threshold
            if peak >= threshold or rolling >= threshold * 0.85:
                return advance_strategy(current)

        if (
            len(reports) >= self._config.stagnation_iterations
            and self._exam_stagnation(reports)
            and current != LearningStrategy.RESEARCH
        ):
            return advance_strategy(current)

        return None

    def _retention_allows_advance(self, retention: RetentionContext) -> bool:
        """Block strategy advance when foundation is unstable."""
        if not self._config.require_retention_stable:
            return True
        if not retention.retention_stable:
            return False
        if self._config.block_advance_on_forgetting and retention.forgetting_reports:
            for report in retention.forgetting_reports:
                if report.regressions:
                    return False
        return True

    def _should_halt_static_illusion(
        self,
        iteration: int,
        kel: KnowledgeEvaluationLayer,
        reports: list[IterationReport],
        retention: RetentionContext | None = None,
    ) -> bool:
        """Avoid halting short noisy runs before strategy phases can play out."""
        if self._should_repair_instead_of_halt(retention):
            return False
        if not self._config.halt_on_static_illusion:
            return False
        if iteration < self._config.min_iterations_before_halt:
            return False
        if len(kel.history("lg")) < self._config.min_lg_history_for_halt:
            return False
        scores = self._recent_scores(reports)
        if scores and max(scores) >= self._config.strategy_advance_threshold * 0.8:
            return False
        return True

    def _should_repair_instead_of_halt(self, retention: RetentionContext | None) -> bool:
        """Prefer maintenance/review over halting when KS is low (M40)."""
        if os.environ.get("ALLM_KS_REPAIR_HALT", "1") != "1":
            return False
        if retention is None or retention.ks is None:
            return False
        threshold = float(os.environ.get("ALLM_KS_ADVANCE_THRESHOLD", "0.70"))
        return retention.ks < threshold

    def _recent_scores(self, reports: list[IterationReport]) -> list[float]:
        """Held-out scores from the trailing strategy-advance window."""
        window = self._config.strategy_advance_window
        rows = reports[-window:] if window > 0 else reports
        return [row.students[0].score_after for row in rows if row.students]

    def _exam_stagnation(self, reports: list[IterationReport]) -> bool:
        """True when held-out-style scores flatline at zero."""
        if len(reports) < 2:
            return False
        recent = [row.students[0].score_after for row in reports[-2:] if row.students]
        return len(recent) == 2 and recent[0] == recent[1] == 0.0


def apply_steering(active: LoopConfig, decision: KelSteeringDecision) -> LoopConfig:
    """Merge a steering decision into the active loop configuration."""
    updates: dict = {}
    if decision.questions_per_exam is not None:
        updates["questions_per_exam"] = decision.questions_per_exam
    if decision.samples_per_iteration is not None:
        updates["samples_per_iteration"] = decision.samples_per_iteration
    if decision.max_goals is not None:
        updates["max_goals"] = decision.max_goals
    if decision.strategy is not None:
        profile = profile_for(decision.strategy)
        updates["strategy"] = decision.strategy.value
        updates["sample_kinds"] = profile.sample_kinds
        updates["use_exam_paraphrase"] = profile.use_exam_paraphrase
        updates["study_failures"] = profile.study_failures
    return active.model_copy(update=updates) if updates else active
