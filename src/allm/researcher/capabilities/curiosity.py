"""Observe phase — measurable curiosity signals."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.capabilities.economy import ProviderReputationLedger
from allm.researcher.multimodal import unsynced_video_gap

logger = get_logger("researcher.curiosity")


class CuriositySignal(BaseModel):
    """One proactive research question with urgency score."""

    model_config = ConfigDict(frozen=True)

    question: str
    topic: str = ""
    score: float = Field(ge=0.0, le=1.0)
    source: str = "curiosity"


class CuriosityReport(BaseModel):
    """Ranked curiosity signals for planning."""

    model_config = ConfigDict(frozen=True)

    signals: tuple[CuriositySignal, ...] = ()
    aggregate_uncertainty: float = 0.0


def _unexplored_providers(store, configured: set[str]) -> list[str]:
    ledger = ProviderReputationLedger(store)
    seen = {row.provider_id for row in ledger.leaderboard()}
    return sorted(configured - seen)


def build_curiosity_report(ctx: CapabilityContext) -> CuriosityReport:
    """Score proactive questions from ecosystem, graph, and provider state."""
    signals: list[CuriositySignal] = []
    cfg = ctx.config

    ecosystem = ctx.ecosystem
    if ecosystem is not None:
        missing = float(getattr(ecosystem, "missing_knowledge", 0))
        if missing > 0.3:
            signals.append(
                CuriositySignal(
                    question="Which recommended topics are not yet in the graph?",
                    topic=cfg.workshop_curriculum_topic,
                    score=round(min(1.0, missing), 4),
                    source="missing_knowledge",
                )
            )
        conflicts = float(getattr(ecosystem, "high_conflict_areas", 0))
        if conflicts > 0.2:
            signals.append(
                CuriositySignal(
                    question="Which concepts have conflicting evidence?",
                    score=round(min(1.0, conflicts), 4),
                    source="high_conflict",
                )
            )
        emerging = int(getattr(ecosystem, "emerging_topics", 0))
        if emerging > 0:
            signals.append(
                CuriositySignal(
                    question="Which emerging topics are growing fastest?",
                    score=round(min(1.0, emerging / 10.0), 4),
                    source="emerging_topics",
                )
            )

    if ctx.state is not None and ctx.student_ids:
        threshold = cfg.mastery_threshold
        waiting = 0
        for student_id in ctx.student_ids:
            topics = ctx.state.topics(student_id)
            for topic in topics:
                confidence = ctx.state.confidence(student_id, topic)
                if confidence is not None and confidence < threshold:
                    waiting += 1
        if waiting:
            signals.append(
                CuriositySignal(
                    question="Which students have been waiting longest for new material?",
                    score=round(min(1.0, waiting / max(1, len(ctx.student_ids) * 3)), 4),
                    source="student_waiting",
                )
            )

    configured = set()
    if cfg.workshop_dir is not None:
        configured.add("kids-workshops")
    if cfg.software_samples is not None:
        configured.add("software-fixture")
    for provider_id in _unexplored_providers(ctx.store, configured):
        signals.append(
            CuriositySignal(
                question=f"Which knowledge does provider {provider_id} offer?",
                score=0.55,
                source="unexplored_provider",
            )
        )

    for finding in ctx.kel_findings:
        mode = getattr(finding, "mode", "")
        if mode == "research_gap":
            signals.append(
                CuriositySignal(
                    question="What knowledge is the ecosystem missing?",
                    score=0.8,
                    source="kel_research_gap",
                )
            )

    if cfg.workshop_dir is not None and cfg.workshop_dir.is_dir():
        mentions, gap = unsynced_video_gap(cfg.workshop_dir, cfg.video_fixture_dir)
        if mentions > 0 and gap > 0:
            signals.append(
                CuriositySignal(
                    question="Which video moments lack synchronized visual evidence?",
                    topic=cfg.workshop_curriculum_topic,
                    score=round(min(1.0, gap / max(1, mentions)), 4),
                    source="unsynced_video",
                )
            )

    ordered = sorted(signals, key=lambda row: -row.score)
    aggregate = round(sum(row.score for row in ordered) / max(1, len(ordered)), 4) if ordered else 0.0
    return CuriosityReport(signals=tuple(ordered), aggregate_uncertainty=aggregate)


class ObserveCuriosityCapability:
    """Observe — rank proactive curiosity questions before planning."""

    level = 0
    name = "observe.curiosity"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        report = build_curiosity_report(ctx)
        pipeline.curiosity_signals = list(report.signals)
        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "observe.curiosity: signals=%d uncertainty=%.2f",
            len(report.signals),
            report.aggregate_uncertainty,
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(report.signals),
                duration_ms=round(elapsed, 2),
                notes=f"uncertainty={report.aggregate_uncertainty:.2f}",
            ),
            artifacts={"curiosity": report},
        )
