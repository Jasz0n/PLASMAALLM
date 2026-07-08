"""Teacher-mediated specialist consultation (ResearcherPlan §Student Consultation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from allm.data.base import Sample
from allm.exam.base import Exam, Question
from allm.exam.grading import Grader
from allm.students.base import Student
from allm.teacher.consultation import request_consultation
from allm.teacher.state import KnowledgeState

if TYPE_CHECKING:
    from allm.researcher.evidence_broker import EvidenceBroker
    from allm.teacher.show_me import ConsultationEvidence


class MediatedConsultationResult(BaseModel):
    """Outcome of Teacher-validated peer teaching."""

    model_config = ConfigDict(frozen=True)

    asker_id: str
    expert_id: str | None
    topic: str
    approved: bool
    expert_answer: str = ""
    expected: str = ""
    reason: str = ""
    study_sample: Sample | None = None
    show_me_requested: bool = False
    evidence: object | None = None


def mediated_consultation(
    state: KnowledgeState,
    grader: Grader,
    asker_id: str,
    expert: Student,
    *,
    topic: str,
    prompt: str,
    expected: str,
    min_expert_confidence: float = 0.3,
    evidence_broker: EvidenceBroker | None = None,
    show_me_query: str | None = None,
    show_me_on_reject: bool = False,
) -> MediatedConsultationResult:
    """Specialist explains; Teacher grades; asker only sees approved material."""
    request = request_consultation(state, asker_id, topic, min_confidence=min_expert_confidence)
    if request.expert_id is None or request.expert_id != expert.student_id:
        return MediatedConsultationResult(
            asker_id=asker_id,
            expert_id=request.expert_id,
            topic=topic,
            approved=False,
            reason=request.reason or "no expert available",
        )

    question = Question(
        id=f"consult-{asker_id}-{topic}",
        prompt=prompt,
        expected=expected,
        topic=topic,
        kind="factual",
    )
    answer = expert.solve(question)
    grade = grader.grade(question, answer)
    approved = grade.correct

    study_sample = None
    if approved and expected:
        study_sample = Sample(
            id=f"approved-{asker_id}-{topic}",
            input=prompt,
            target=expected,
            metadata={"topic": topic, "sample_kind": "consultation", "expert": expert.student_id},
        )

    evidence = None
    show_me_requested = False
    if evidence_broker is not None:
        from allm.teacher.show_me import consultation_show_me

        should_show = show_me_query is not None or (show_me_on_reject and not approved)
        if should_show:
            show_me_requested = True
            evidence = consultation_show_me(
                evidence_broker,
                asker_id=asker_id,
                topic=topic,
                prompt=prompt,
                query=show_me_query,
            )
            if evidence.found and not approved:
                reason = "teacher rejected explanation; visual evidence provided"
            elif evidence.found:
                reason = "teacher approved with supporting visual evidence"
            else:
                reason = (
                    "teacher approved specialist explanation"
                    if approved
                    else "teacher rejected explanation; no visual evidence found"
                )
        else:
            reason = (
                "teacher approved specialist explanation"
                if approved
                else "teacher rejected explanation"
            )
    else:
        reason = (
            "teacher approved specialist explanation"
            if approved
            else "teacher rejected explanation"
        )

    return MediatedConsultationResult(
        asker_id=asker_id,
        expert_id=expert.student_id,
        topic=topic,
        approved=approved,
        expert_answer=answer.text,
        expected=expected,
        reason=reason,
        study_sample=study_sample,
        show_me_requested=show_me_requested,
        evidence=evidence,
    )
