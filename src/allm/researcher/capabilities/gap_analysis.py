"""Graph gap analysis — missing prerequisites and weak links."""

from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.missions import MissionStore

logger = get_logger("researcher.gap")


class GraphGap(BaseModel):
    """A structural hole in the knowledge graph."""

    model_config = ConfigDict(frozen=True)

    parent: str
    child: str
    missing_prerequisite: str
    priority: float = 0.6


class GapAnalysisReport(BaseModel):
    """All detected graph gaps in one observation."""

    model_config = ConfigDict(frozen=True)

    gaps: tuple[GraphGap, ...] = ()


def analyze_graph_gaps(graph) -> GapAnalysisReport:
    """Find prerequisite chains with missing intermediate concepts."""
    if graph is None:
        return GapAnalysisReport()

    names = set(graph.names())
    gaps: list[GraphGap] = []

    for concept in graph.concepts():
        if concept.status != "active":
            continue
        for prereq in concept.prerequisites:
            if prereq not in names:
                gaps.append(
                    GraphGap(
                        parent=concept.name,
                        child=concept.name,
                        missing_prerequisite=prereq,
                        priority=0.75,
                    )
                )
            else:
                prereq_node = graph.get(prereq)
                if prereq_node is None:
                    continue
                for sub in prereq_node.prerequisites:
                    if sub not in names:
                        gaps.append(
                            GraphGap(
                                parent=prereq,
                                child=concept.name,
                                missing_prerequisite=sub,
                                priority=0.65,
                            )
                        )

    return GapAnalysisReport(gaps=tuple(gaps))


class GraphGapAnalysisCapability:
    """Analyze knowledge graph for missing nodes and weak prerequisite chains."""

    level = 0
    name = "analysis.gap"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        report = analyze_graph_gaps(ctx.graph)
        pipeline.graph_gaps = list(report.gaps)

        store = MissionStore(ctx.store)
        for gap in report.gaps[:5]:
            store.open_from_gap(
                parent=gap.parent,
                child=gap.child,
                missing=gap.missing_prerequisite,
                priority=gap.priority,
            )

        elapsed = (time.perf_counter() - started) * 1000
        logger.info("analysis.gap: gaps=%d", len(report.gaps))
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(report.gaps),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"gaps": report},
        )


class MissionReviewCapability:
    """Load active missions and attach to pipeline state."""

    level = 0
    name = "missions.review"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        store = MissionStore(ctx.store)
        active = store.active()
        pipeline.active_missions = active
        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(active),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"missions": active},
        )
