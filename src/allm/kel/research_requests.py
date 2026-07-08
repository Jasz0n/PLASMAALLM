"""KEL-generated research requests — ask Researcher to improve teaching (M45)."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.kel.types import Finding
from allm.loop.learning_loop import IterationReport
from allm.loop.retention_gates import RetentionContext

TriggerKind = Literal[
    "unstable_mastery",
    "forgetting",
    "repair_mode",
    "strategy_stagnation",
    "high_conflict",
    "research_gap",
]


class KelResearchRequest(BaseModel):
    """Structured task for Researcher when teaching material may be insufficient."""

    model_config = ConfigDict(frozen=True)

    id: str
    topic: str
    task: str
    trigger: TriggerKind
    priority: float = Field(ge=0.0, le=1.0)
    student_id: str | None = None
    search_hints: tuple[str, ...] = ()
    reason: str = ""


def kel_research_requests_enabled() -> bool:
    return os.environ.get("ALLM_KEL_RESEARCH_REQUESTS", "1") == "1"


def strategy_stagnation_iters() -> int:
    return max(2, int(os.environ.get("ALLM_KEL_STRATEGY_STAGNATION", "4")))


def _strategy_stagnation(
    reports: list[IterationReport],
    strategy: str,
    *,
    min_iters: int,
) -> bool:
    if len(reports) < min_iters:
        return False
    recent = reports[-min_iters:]
    for report in recent:
        if not report.students:
            return False
        if report.students[0].strategy != strategy:
            return False
    return True


def build_kel_research_requests(
    *,
    findings: tuple[Finding, ...],
    compromise_mode: str | None,
    retention: RetentionContext | None,
    reports: list[IterationReport],
    student_id: str,
    topics: tuple[str, ...],
    strategy: str,
    kel_ks: float | None,
) -> tuple[KelResearchRequest, ...]:
    """Turn KEL diagnosis into actionable Researcher tasks."""
    if not kel_research_requests_enabled():
        return ()

    primary_topic = topics[0] if topics else "general"
    rows: list[KelResearchRequest] = []
    seen: set[str] = set()

    def add(
        *,
        req_id: str,
        topic: str,
        task: str,
        trigger: TriggerKind,
        priority: float,
        search_hints: tuple[str, ...] = (),
        reason: str,
    ) -> None:
        key = f"{trigger}::{topic}"
        if key in seen:
            return
        seen.add(key)
        rows.append(
            KelResearchRequest(
                id=req_id,
                topic=topic,
                task=task,
                trigger=trigger,
                priority=round(priority, 4),
                student_id=student_id,
                search_hints=search_hints,
                reason=reason,
            )
        )

    if compromise_mode in {"repair", "maintain"}:
        ks_label = f"{kel_ks:.2f}" if kel_ks is not None else "n/a"
        add(
            req_id=f"kel-repair-{student_id}-{primary_topic}",
            topic=primary_topic,
            task=(
                f"Find better explanations, visuals, and worked examples for "
                f"{primary_topic} — student retention unstable (KS={ks_label})"
            ),
            trigger="repair_mode",
            priority=0.92,
            search_hints=("visual", "diagram", "worked_example", "prerequisite"),
            reason=f"KEL compromise mode={compromise_mode}",
        )

    if _strategy_stagnation(reports, strategy, min_iters=strategy_stagnation_iters()):
        add(
            req_id=f"kel-strategy-{student_id}-{strategy}",
            topic=primary_topic,
            task=(
                f"Find application exercises, relations, and non-definition material "
                f"for {primary_topic} — stuck on {strategy} for "
                f"{strategy_stagnation_iters()} iterations"
            ),
            trigger="strategy_stagnation",
            priority=0.88,
            search_hints=("relations", "examples", "application", "experiment"),
            reason=f"strategy {strategy!r} stagnation",
        )

    for finding in findings:
        if finding.mode == "unstable_mastery":
            add(
                req_id=f"kel-unstable-{student_id}-{primary_topic}",
                topic=primary_topic,
                task=(
                    f"Find teaching material that improves retention for {primary_topic} "
                    f"without adding conflicting definitions"
                ),
                trigger="unstable_mastery",
                priority=0.85,
                search_hints=("review", "spaced_repetition", "visual"),
                reason=finding.detail,
            )
        elif finding.mode == "high_conflict_discovery":
            add(
                req_id=f"kel-conflict-{primary_topic}",
                topic=primary_topic,
                task=(
                    f"Classify and reconcile conflicting definitions for {primary_topic} "
                    f"(terminology vs conceptual vs empirical)"
                ),
                trigger="high_conflict",
                priority=0.80,
                search_hints=("conflict_resolution", "cross_source", "terminology"),
                reason=finding.detail,
            )
        elif finding.mode == "research_gap":
            add(
                req_id=f"kel-gap-{primary_topic}",
                topic=primary_topic,
                task=f"Discover missing foundational material for {primary_topic}",
                trigger="research_gap",
                priority=0.75,
                search_hints=("discovery", "prerequisite"),
                reason=finding.detail,
            )

    if retention is not None and retention.forgetting_reports:
        for report in retention.forgetting_reports:
            for topic in report.regressions:
                add(
                    req_id=f"kel-forget-{student_id}-{topic}",
                    topic=topic,
                    task=(
                        f"Find prerequisite and review material — student forgot {topic} "
                        f"after new learning"
                    ),
                    trigger="forgetting",
                    priority=0.90,
                    search_hints=("prerequisite", "review", "visual"),
                    reason=f"forgetting on {topic}",
                )

    rows.sort(key=lambda row: -row.priority)
    limit = int(os.environ.get("ALLM_KEL_RESEARCH_REQUEST_LIMIT", "6"))
    return tuple(rows[:limit])
