"""Capability-driven Researcher pipeline."""

from __future__ import annotations

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    PipelineState,
    persist_capability_metrics,
)
from allm.researcher.capabilities.improvement import ImprovementCapability
from allm.researcher.capabilities.planning import ResearchPlan
from allm.researcher.capabilities.registry import get_capability, pipeline_order
from allm.researcher.queue import RecommendationQueue
from allm.researcher.types import ResearcherReport
from allm.storage.base import RecordStore

logger = get_logger("researcher.pipeline")


class CapabilityPipeline:
    """Orchestrates L0–L7 capabilities into one research cycle."""

    def __init__(self, store: RecordStore, queue: RecommendationQueue) -> None:
        self._store = store
        self._queue = queue

    def run_cycle(self, ctx: CapabilityContext) -> ResearcherReport:
        """Execute enabled capabilities and enqueue outputs."""
        pipeline = PipelineState()
        plan: ResearchPlan | None = ctx.plan if isinstance(ctx.plan, ResearchPlan) else None
        capability_results = []

        order = pipeline_order(
            ctx.config.enabled_capabilities,
            discovery_order=ctx.config.discovery_source_order,
        )
        for name in order:
            capability = get_capability(name)
            result = capability.run(ctx, pipeline)
            persist_capability_metrics(self._store, result)
            capability_results.append(result)
            if name == "planning.research" and "plan" in result.artifacts:
                plan = result.artifacts["plan"]

        packages = pipeline.verified_packages or pipeline.packages
        for package in packages:
            self._queue.store_package(package)
        for rec in pipeline.recommendations:
            self._queue.enqueue(rec)

        strategy_hints = None
        curiosity_signals = ()
        graph_gaps = ()
        active_missions = ()
        multimodal_synced = ()
        for result in capability_results:
            if result.capability == "improvement.reflect":
                strategy_hints = result.artifacts.get("strategy_hints")
            if result.capability == "observe.curiosity" and "curiosity" in result.artifacts:
                curiosity_signals = result.artifacts["curiosity"].signals
            if result.capability == "analysis.gap" and "gaps" in result.artifacts:
                graph_gaps = result.artifacts["gaps"].gaps
            if result.capability == "missions.review":
                active_missions = tuple(result.artifacts.get("missions", ()))
            if result.capability == "understanding.sync" and "synced" in result.artifacts:
                multimodal_synced = tuple(result.artifacts["synced"])

        if pipeline.multimodal_synced and len(pipeline.multimodal_synced) >= len(multimodal_synced):
            multimodal_synced = tuple(pipeline.multimodal_synced)

        logger.info(
            "pipeline: plan=%r packages=%d recommendations=%d capabilities=%d",
            plan.goal if plan else None,
            len(packages),
            len(pipeline.recommendations),
            len(capability_results),
        )

        return ResearcherReport(
            packages=tuple(packages),
            recommendations=tuple(pipeline.recommendations),
            providers_evaluated=pipeline.providers_evaluated,
            conflicts_detected=pipeline.conflicts_detected,
            plan=plan,
            proposal_hints=tuple(pipeline.proposal_hints),
            capability_summary=tuple(
                (r.capability, r.metrics.yield_count, r.metrics.notes) for r in capability_results
            ),
            strategy_hints=strategy_hints,
            curiosity_signals=curiosity_signals,
            graph_gaps=graph_gaps,
            active_missions=active_missions,
            multimodal_synced=multimodal_synced,
            cross_source_report=pipeline.cross_source_report,
        )

    @staticmethod
    def load_strategy_hints(store: RecordStore):
        """Strategy hints from the last improvement capability run."""
        return ImprovementCapability.load_strategy_hints(store)
