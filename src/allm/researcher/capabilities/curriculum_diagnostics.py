"""L4 — Curriculum diagnostics for KEL remediation requests (M46)."""

from __future__ import annotations

import json
import time

from allm.core.logging import get_logger
from allm.kel.research_requests import KelResearchRequest
from allm.loop.history import LearningIterationRecord
from allm.models import load_model
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.curriculum_diagnostics import (
    DiagnosticContext,
    curriculum_diagnostics_enabled,
    diagnose_requests,
    format_diagnostic,
    reasoning_diagnostics_enabled,
)
from allm.researcher.model_router import resolve_model_spec, route_request
from allm.researcher.remediation import requests_to_recommendations

logger = get_logger("researcher.curriculum_diagnostics")

LEARNING_HISTORY_NAMESPACE = "learning_history"
KEL_REQUESTS_NAMESPACE = "kel_research_requests"


def _load_kel_requests(ctx: CapabilityContext) -> tuple[KelResearchRequest, ...]:
    rows: list[KelResearchRequest] = []
    for key in ctx.store.keys(KEL_REQUESTS_NAMESPACE):
        record = ctx.store.get(KEL_REQUESTS_NAMESPACE, key)
        if record is not None:
            rows.append(KelResearchRequest.model_validate(record.value))
    rows.sort(key=lambda row: -row.priority)
    return tuple(rows)


def _load_history(ctx: CapabilityContext) -> tuple[LearningIterationRecord, ...]:
    rows: list[LearningIterationRecord] = []
    for key in ctx.store.keys(LEARNING_HISTORY_NAMESPACE):
        record = ctx.store.get(LEARNING_HISTORY_NAMESPACE, key)
        if record is None:
            continue
        rows.append(LearningIterationRecord.model_validate(record.value))
    rows.sort(key=lambda row: row.iteration)
    return tuple(rows)


def _conflict_count(pipeline: PipelineState) -> int:
    total = pipeline.conflicts_detected
    if total:
        return total
    return sum(len(package.conflicts) for package in pipeline.packages)


class CurriculumDiagnosticsCapability:
    """L4 — diagnose why KEL remediation is needed before targeting curriculum."""

    level = 4
    name = "diagnostics.curriculum"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        if not curriculum_diagnostics_enabled():
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="curriculum diagnostics disabled",
                ),
            )

        requests = _load_kel_requests(ctx)
        if not requests:
            return CapabilityResult(
                capability=self.name,
                metrics=CapabilityMetrics(
                    capability=self.name,
                    level=self.level,
                    yield_count=0,
                    notes="no kel research requests",
                ),
            )

        history = _load_history(ctx)
        latest = history[-1] if history else None
        context = DiagnosticContext(
            failure_prompts=latest.failure_prompts if latest else (),
            strategy=latest.strategy if latest else None,
            kel_ks=(latest.metadata or {}).get("kel_ks") if latest else None,
            conflict_count=_conflict_count(pipeline),
            history=history,
        )

        model = None
        if reasoning_diagnostics_enabled():
            role = route_request(requests[0])
            spec = resolve_model_spec(role)
            if role != "vision":
                try:
                    model = load_model(spec)
                except (OSError, RuntimeError):
                    model = None

        diagnostics = diagnose_requests(requests, context, model=model)
        pipeline.curriculum_diagnostics = list(diagnostics)
        remediation = requests_to_recommendations(requests, diagnostics=diagnostics)
        pipeline.recommendations.extend(remediation)

        for diagnostic in diagnostics[:3]:
            logger.info("diagnostics.curriculum: %s", format_diagnostic(diagnostic))

        for diagnostic in diagnostics:
            ctx.store.put(
                "curriculum_diagnostics",
                diagnostic.request_id,
                json.loads(diagnostic.model_dump_json()),
                reason="curriculum diagnostic",
            )

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "diagnostics.curriculum: hypotheses=%d remediation=%d",
            len(diagnostics),
            len(remediation),
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(diagnostics),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"diagnostics": diagnostics, "remediation": remediation},
        )
