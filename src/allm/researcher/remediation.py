"""Remediation recommendations from KEL research requests (M45) and diagnostics (M46)."""

from __future__ import annotations

from allm.kel.research_requests import KelResearchRequest
from allm.researcher.curriculum_diagnostics import CurriculumDiagnostic
from allm.researcher.types import ResearchRecommendation


def requests_to_recommendations(
    requests: tuple[KelResearchRequest, ...] | list[KelResearchRequest],
    *,
    diagnostics: tuple[CurriculumDiagnostic, ...] = (),
) -> list[ResearchRecommendation]:
    """Convert KEL research tasks into Researcher remediation recommendations."""
    diagnostic_by_id = {row.request_id: row for row in diagnostics}
    rows: list[ResearchRecommendation] = []
    for request in requests:
        hints = ", ".join(request.search_hints) if request.search_hints else ""
        proposal = request.task
        diagnostic = diagnostic_by_id.get(request.id)
        if diagnostic is not None:
            rec_text = "; ".join(diagnostic.recommendations[:2])
            proposal = (
                f"{request.task} | Diagnosis: {diagnostic.failure_reason} "
                f"({diagnostic.confidence:.2f}) — {rec_text}"
            )
        if hints:
            proposal = f"{proposal} [hints: {hints}]"
        reason = f"KEL remediation: {request.reason}"
        if diagnostic is not None and diagnostic.evidence:
            reason += f" | evidence: {diagnostic.evidence[0]}"
        rows.append(
            ResearchRecommendation(
                topic=request.topic,
                priority=request.priority,
                reason=reason,
                package_id=f"kel-request::{request.id}",
                provider="kel-research",
                concept=request.topic,
                suggested_students=(request.student_id,) if request.student_id else (),
                proposal_hint=proposal,
                recommendation_kind="remediation",
                knowledge_tier="established",
            )
        )
    return rows
