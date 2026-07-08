"""Curriculum diagnostics — why learning strategies fail (M46)."""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.evaluation.strategy_gain import StrategyPhaseGain, compute_marginal_strategy_gains
from allm.kel.research_requests import KelResearchRequest, TriggerKind
from allm.loop.history import LearningIterationRecord
from allm.models.base import LanguageModel, ModelSpec
from allm.researcher.model_router import SpecialistRole, model_router_enabled, resolve_model_spec, route_request

FailureReason = Literal[
    "missing_prerequisite",
    "weak_relation_material",
    "conflicting_sources",
    "retention_instability",
    "curriculum_ceiling",
    "insufficient_visuals",
    "unknown",
]

_REASON = re.compile(
    r"^\s*FAILURE_REASON:\s*(\w+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_CONFIDENCE = re.compile(
    r"^\s*CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_EVIDENCE = re.compile(
    r"^\s*EVIDENCE:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_RECOMMENDATION = re.compile(
    r"^\s*RECOMMENDATION:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class DiagnosticContext(BaseModel):
    """Signals available when KEL asks Researcher to investigate."""

    model_config = ConfigDict(frozen=True)

    failure_prompts: tuple[str, ...] = ()
    strategy: str | None = None
    kel_ks: float | None = None
    conflict_count: int | None = None
    history: tuple[LearningIterationRecord, ...] = ()


class CurriculumDiagnostic(BaseModel):
    """Structured hypothesis about why teaching is not working."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    topic: str
    trigger: TriggerKind
    failure_reason: FailureReason
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    specialist_role: SpecialistRole = "reasoning"
    model_id: str = "heuristic"


def curriculum_diagnostics_enabled() -> bool:
    return os.environ.get("ALLM_CURRICULUM_DIAGNOSTICS", "1") == "1"


def reasoning_diagnostics_enabled() -> bool:
    backend = os.environ.get("ALLM_CURRICULUM_DIAGNOSTICS_BACKEND", "auto").lower()
    if backend in {"heuristic", "stub", "offline"}:
        return False
    if backend == "reasoning":
        return True
    return backend in {"auto", "hybrid"}


def _phase_lookup(
    history: tuple[LearningIterationRecord, ...],
) -> dict[str, StrategyPhaseGain]:
    if not history:
        return {}
    phases = compute_marginal_strategy_gains(list(history))
    return {phase.strategy: phase for phase in phases}


def _definition_style_failures(prompts: tuple[str, ...]) -> int:
    return sum(1 for prompt in prompts if prompt.lower().startswith("what is"))


def diagnose_heuristic(
    request: KelResearchRequest,
    context: DiagnosticContext,
) -> CurriculumDiagnostic:
    """Rule-based curriculum diagnosis without LLM calls."""
    role = route_request(request)
    spec = resolve_model_spec(role)
    phases = _phase_lookup(context.history)
    evidence: list[str] = []
    recommendations: list[str] = []
    reason: FailureReason = "unknown"
    confidence = 0.55

    relations = phases.get("relations")
    definitions = phases.get("definitions")
    if relations is not None and definitions is not None:
        if relations.kel_lg_delta is not None and definitions.kel_lg_delta is not None:
            if relations.kel_lg_delta < definitions.kel_lg_delta * 0.25:
                evidence.append(
                    "relations KEL LG delta is much lower than definitions phase"
                )
                recommendations.extend(
                    [
                        "Create prerequisite definition review before relation exercises",
                        "Add visual dependency graph for concept links",
                    ]
                )
                reason = "weak_relation_material"
                confidence = 0.78

    if request.trigger == "high_conflict" or (context.conflict_count or 0) > 100:
        reason = "conflicting_sources"
        confidence = max(confidence, 0.82)
        evidence.append(
            f"{context.conflict_count or 'many'} preserved package conflict(s) detected"
        )
        recommendations.extend(
            [
                "Reconcile terminology conflicts before adding new lessons",
                "Route disputed claims to verifier specialist",
            ]
        )

    if request.trigger in {"repair_mode", "unstable_mastery", "forgetting"}:
        if context.kel_ks is not None and context.kel_ks < 0.70:
            evidence.append(f"KEL KS {context.kel_ks:.2f} below retention threshold")
            recommendations.append("Increase spaced review and reduce new concept load")
            if reason == "unknown":
                reason = "retention_instability"
                confidence = 0.74

    if request.trigger == "strategy_stagnation" and context.strategy == "definitions":
        if definitions is not None and definitions.heldout_gain <= 0:
            reason = "curriculum_ceiling"
            confidence = max(confidence, 0.71)
            evidence.append("definitions strategy stagnated without held-out gain")
            recommendations.extend(
                [
                    "Introduce transfer exercises beyond verbatim definitions",
                    "Add visual worked examples for holdout concepts",
                ]
            )

    if _definition_style_failures(context.failure_prompts) >= 3:
        evidence.append(
            f"{_definition_style_failures(context.failure_prompts)} failures are definition-style prompts"
        )
        recommendations.append("Teach missing definition prerequisites before relations")
        if reason in {"unknown", "weak_relation_material"}:
            reason = "missing_prerequisite"
            confidence = max(confidence, 0.80)

    if "visual" in request.search_hints and reason == "unknown":
        reason = "insufficient_visuals"
        confidence = 0.68
        evidence.append("KEL requested visual remediation hints")
        recommendations.append("Commission diagram-first lesson from vision specialist")

    if not evidence:
        evidence = (request.reason or request.task,)[:1]
    if not recommendations:
        recommendations = [request.task]

    return CurriculumDiagnostic(
        request_id=request.id,
        topic=request.topic,
        trigger=request.trigger,
        failure_reason=reason,
        confidence=round(confidence, 4),
        evidence=tuple(evidence[:6]),
        recommendations=tuple(dict.fromkeys(recommendations[:5])),
        specialist_role=role,
        model_id=spec.model_id if model_router_enabled() else "heuristic",
    )


def build_reasoning_prompt(
    request: KelResearchRequest,
    context: DiagnosticContext,
) -> str:
    """Prompt for the reasoning specialist to diagnose curriculum failure."""
    failures = "\n".join(f"- {prompt}" for prompt in context.failure_prompts[:8]) or "- (none)"
    phases = _phase_lookup(context.history)
    phase_lines = []
    for strategy, phase in phases.items():
        kel = f"{phase.kel_lg_delta:+.3f}" if phase.kel_lg_delta is not None else "n/a"
        phase_lines.append(
            f"- {strategy}: held-out {phase.heldout_gain:+.2f}, KEL LG {kel}"
        )
    phase_text = "\n".join(phase_lines) or "- (no history)"
    return (
        "You are the Chief Scientist diagnosing why a student learning loop is failing.\n"
        "Reply with exactly these lines:\n"
        "FAILURE_REASON: missing_prerequisite|weak_relation_material|conflicting_sources|"
        "retention_instability|curriculum_ceiling|insufficient_visuals|unknown\n"
        "CONFIDENCE: 0.0-1.0\n"
        "EVIDENCE: one short sentence\n"
        "RECOMMENDATION: one actionable teaching fix\n\n"
        f"Topic: {request.topic}\n"
        f"Trigger: {request.trigger}\n"
        f"Task: {request.task}\n"
        f"Strategy: {context.strategy or 'unknown'}\n"
        f"KEL KS: {context.kel_ks if context.kel_ks is not None else 'n/a'}\n"
        f"Conflicts: {context.conflict_count if context.conflict_count is not None else 'n/a'}\n"
        f"Failed exam prompts:\n{failures}\n"
        f"Strategy phases:\n{phase_text}\n"
    )


def parse_reasoning_response(raw: str) -> tuple[FailureReason, float, tuple[str, ...], tuple[str, ...]]:
    """Parse structured reasoning-model diagnostic output."""
    reason_match = _REASON.search(raw)
    reason_text = reason_match.group(1).lower() if reason_match else "unknown"
    allowed = {
        "missing_prerequisite",
        "weak_relation_material",
        "conflicting_sources",
        "retention_instability",
        "curriculum_ceiling",
        "insufficient_visuals",
        "unknown",
    }
    failure_reason: FailureReason = reason_text if reason_text in allowed else "unknown"  # type: ignore[assignment]
    conf_match = _CONFIDENCE.search(raw)
    confidence = float(conf_match.group(1)) if conf_match else 0.6
    evidence = tuple(match.group(1).strip() for match in _EVIDENCE.finditer(raw))[:4]
    recommendations = tuple(match.group(1).strip() for match in _RECOMMENDATION.finditer(raw))[:4]
    return failure_reason, min(1.0, max(0.0, confidence)), evidence, recommendations


def diagnose_with_reasoning(
    request: KelResearchRequest,
    context: DiagnosticContext,
    model: LanguageModel,
    *,
    spec: ModelSpec,
) -> CurriculumDiagnostic:
    """Run curriculum diagnostics through a reasoning specialist model."""
    prompt = build_reasoning_prompt(request, context)
    try:
        raw = model.generate(prompt, spec.generation)
        reason, confidence, evidence, recommendations = parse_reasoning_response(raw)
        role = route_request(request)
        if not evidence:
            evidence = (raw.strip()[:160],)
        if not recommendations:
            recommendations = (request.task,)
        return CurriculumDiagnostic(
            request_id=request.id,
            topic=request.topic,
            trigger=request.trigger,
            failure_reason=reason,
            confidence=round(confidence, 4),
            evidence=evidence,
            recommendations=recommendations,
            specialist_role=role,
            model_id=spec.model_id,
        )
    except (OSError, RuntimeError, ValueError):
        return diagnose_heuristic(request, context)


def diagnose_request(
    request: KelResearchRequest,
    context: DiagnosticContext,
    *,
    model: LanguageModel | None = None,
) -> CurriculumDiagnostic:
    """Diagnose one KEL research request using heuristics and optional reasoning."""
    if reasoning_diagnostics_enabled() and model is not None:
        role = route_request(request)
        spec = resolve_model_spec(role)
        if role != "vision":
            return diagnose_with_reasoning(request, context, model, spec=spec)
    return diagnose_heuristic(request, context)


def diagnose_requests(
    requests: tuple[KelResearchRequest, ...] | list[KelResearchRequest],
    context: DiagnosticContext,
    *,
    model: LanguageModel | None = None,
) -> tuple[CurriculumDiagnostic, ...]:
    """Diagnose a batch of KEL research requests."""
    if not curriculum_diagnostics_enabled():
        return ()
    return tuple(diagnose_request(request, context, model=model) for request in requests)


def format_diagnostic(diagnostic: CurriculumDiagnostic) -> str:
    """Single-line summary for logs."""
    evidence = diagnostic.evidence[0] if diagnostic.evidence else "n/a"
    return (
        f"{diagnostic.failure_reason} ({diagnostic.confidence:.2f}) "
        f"via {diagnostic.specialist_role}/{diagnostic.model_id}: {evidence}"
    )
