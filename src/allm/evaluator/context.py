"""Build evaluation inputs from loop state (M43)."""

from __future__ import annotations

from allm.evaluator.types import EvaluationInput
from allm.exam.verdicts import mean_alignment_score, mean_curriculum_score, mean_evidence_score
from allm.kel.layer import KnowledgeEvaluationLayer
from allm.loop.learning_loop import IterationReport
from allm.loop.maintenance_curriculum import maintenance_split_from_env
from allm.loop.retention_gates import RetentionContext
from allm.teacher.state import KnowledgeState


def build_evaluation_input(
    *,
    state: KnowledgeState,
    student_id: str,
    topics: tuple[str, ...],
    reports: list[IterationReport],
    kel: KnowledgeEvaluationLayer,
    retention: RetentionContext | None,
    kel_ks: float | None,
) -> EvaluationInput:
    """Assemble independent evaluator inputs from KEL and loop history."""
    first = reports[0].students[0].score_before if reports and reports[0].students else None
    last = reports[-1].students[0].score_after if reports and reports[-1].students else None
    peak = retention.heldout_peak if retention is not None else None
    if peak is None and reports:
        peak = max(
            (row.students[0].score_after for row in reports if row.students),
            default=None,
        )

    split = maintenance_split_from_env(kel_ks)
    review_fraction = split.review_fraction if split is not None else 0.0

    ecosystem = kel._last_ecosystem()  # noqa: SLF001 — measurement bridge
    debate = reports[-1].debate_disagreement if reports else None

    curriculum_score = _recent_curriculum_score(state, student_id)
    alignment_score = _recent_alignment_score(state, student_id)
    evidence_score = _recent_evidence_score(state, student_id)

    return EvaluationInput(
        student_id=student_id,
        topics=topics,
        heldout_first=first,
        heldout_last=last,
        heldout_peak=peak,
        kel_lg=kel._last("lg"),  # noqa: SLF001
        kel_ks=kel_ks,
        kel_cd=kel._last("cd"),  # noqa: SLF001
        kel_cre=kel._last("cre"),  # noqa: SLF001
        debate_disagreement=debate,
        review_fraction=review_fraction,
        mean_forgetting_risk=0.0,
        curriculum_score=curriculum_score,
        alignment_score=alignment_score,
        evidence_score=evidence_score,
        missing_knowledge=getattr(ecosystem, "missing_knowledge", None) if ecosystem else None,
        conflict_discovery=getattr(ecosystem, "high_conflict_areas", None) if ecosystem else None,
    )


def _recent_curriculum_score(state: KnowledgeState, student_id: str) -> float | None:
    exams = state.exam_results(student_id)[-3:]
    scores: list[float] = []
    for exam in exams:
        value = mean_curriculum_score(exam.results)
        if value is not None:
            scores.append(value)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _recent_alignment_score(state: KnowledgeState, student_id: str) -> float | None:
    exams = state.exam_results(student_id)[-3:]
    scores: list[float] = []
    for exam in exams:
        value = mean_alignment_score(exam.results)
        if value is not None:
            scores.append(value)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _recent_evidence_score(state: KnowledgeState, student_id: str) -> float | None:
    exams = state.exam_results(student_id)[-3:]
    scores: list[float] = []
    for exam in exams:
        value = mean_evidence_score(exam.results)
        if value is not None:
            scores.append(value)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)
