"""L0 — Research planning before discovery."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)


class PlanStep(BaseModel):
    """One ordered step in a research plan."""

    model_config = ConfigDict(frozen=True)

    capability: str
    detail: str = ""


class ResearchPlan(BaseModel):
    """What the ecosystem should acquire next."""

    model_config = ConfigDict(frozen=True)

    goal: str
    steps: tuple[PlanStep, ...] = ()
    target_topics: tuple[str, ...] = ()
    target_providers: tuple[str, ...] = ()
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class StrategyHints(BaseModel):
    """L7 feedback consumed by the next planning cycle."""

    model_config = ConfigDict(frozen=True)

    skip_providers: tuple[str, ...] = ()
    prefer_providers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def build_research_plan(ctx: CapabilityContext, pipeline: PipelineState | None = None) -> ResearchPlan:
    """Heuristic plan from curiosity, gaps, missions, KEL, and ecosystem."""
    pipe = pipeline or PipelineState()
    steps: list[PlanStep] = []
    target_topics: set[str] = set()
    target_providers: set[str] = set()
    goal_parts: list[str] = []
    priority = 0.5

    for signal in pipe.curiosity_signals[:3]:
        goal_parts.append(signal.question)
        priority = max(priority, signal.score)
        if signal.topic:
            target_topics.add(signal.topic)

    for gap in pipe.graph_gaps[:3]:
        goal_parts.append(f"fill gap {gap.missing_prerequisite}")
        target_topics.add(gap.missing_prerequisite)
        priority = max(priority, gap.priority)

    for mission in pipe.active_missions[:3]:
        goal_parts.append(mission.goal)
        target_topics.update(mission.target_topics)
        priority = max(priority, mission.priority)
        steps.append(PlanStep(capability="missions.review", detail=f"mission {mission.id}"))

    hints = ctx.strategy_hints
    if hints is not None and hasattr(hints, "skip_providers"):
        skip = set(hints.skip_providers)
    else:
        skip = set()

    cfg = ctx.config
    if cfg.workshop_dir is not None and "kids-workshops" not in skip:
        target_providers.add("kids-workshops")
        steps.append(PlanStep(capability="discovery.workshop", detail="distill workshop transcripts"))

    if cfg.software_samples is not None and "software-fixture" not in skip:
        target_providers.add("software-fixture")
        steps.append(PlanStep(capability="discovery.software", detail="load software fixture"))

    ecosystem = ctx.ecosystem
    if ecosystem is not None:
        if getattr(ecosystem, "missing_knowledge", 0) >= 0.4:
            goal_parts.append("fill knowledge gaps")
            priority = max(priority, 0.7)
            if cfg.workshop_curriculum_topic:
                target_topics.add(cfg.workshop_curriculum_topic)
        if getattr(ecosystem, "high_conflict_areas", 0) >= 0.35:
            goal_parts.append("verify conflicts")
            steps.append(PlanStep(capability="verification.graph", detail="compare with graph"))
            priority = max(priority, 0.65)

    for finding in ctx.kel_findings:
        mode = getattr(finding, "mode", "")
        if mode == "research_gap":
            goal_parts.append("address research gap")
            priority = max(priority, 0.75)
        if mode == "high_conflict_discovery":
            steps.append(PlanStep(capability="verification.graph", detail="conflict verification"))

    if not steps:
        steps = [
            PlanStep(capability="discovery.workshop", detail="default workshop scan"),
            PlanStep(capability="discovery.software", detail="default software scan"),
        ]

    steps.extend(
        [
            PlanStep(capability="understanding.package", detail="build knowledge packages"),
            PlanStep(capability="verification.graph", detail="verify against graph"),
            PlanStep(capability="curriculum.target", detail="target specialists"),
            PlanStep(capability="ecosystem.analyze", detail="ecosystem snapshot"),
            PlanStep(capability="economy.ledger", detail="update provider ledger"),
            PlanStep(capability="improvement.reflect", detail="record capability yields"),
        ]
    )

    goal = "; ".join(goal_parts) if goal_parts else "continuous ecosystem learning"
    return ResearchPlan(
        goal=goal,
        steps=tuple(steps),
        target_topics=tuple(sorted(target_topics)),
        target_providers=tuple(sorted(target_providers)),
        priority=round(priority, 4),
    )


class PlanCapability:
    """L0 — create a research plan before discovery."""

    level = 0
    name = "planning.research"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        plan = ctx.plan if ctx.plan is not None else build_research_plan(ctx, pipeline)
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(plan.steps),
                notes=plan.goal,
            ),
            artifacts={"plan": plan},
        )
