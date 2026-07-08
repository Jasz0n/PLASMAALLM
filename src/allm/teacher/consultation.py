"""Peer consultation: ask the best domain expert."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from allm.collector.pool import SamplePool
from allm.data.base import Sample
from allm.exam.base import ExamResult
from allm.exam.grading import Grader
from allm.students.base import Student
from allm.students.identity import StudentIdentity, domain_fit
from allm.teacher.experts import best_expert
from allm.teacher.state import KnowledgeState

if TYPE_CHECKING:
    from allm.teacher.mediated_consultation import MediatedConsultationResult


class ConsultationRequest(BaseModel):
    """A student asking a specialist for help on a topic."""

    model_config = ConfigDict(frozen=True)

    asker_id: str
    topic: str
    expert_id: str | None
    reason: str


def request_consultation(
    state: KnowledgeState,
    asker_id: str,
    topic: str,
    *,
    min_confidence: float = 0.3,
) -> ConsultationRequest:
    """Pick the best expert for ``topic`` if someone else knows it better."""
    expert_id = best_expert(state, topic, min_confidence=min_confidence)
    if expert_id is None or expert_id == asker_id:
        return ConsultationRequest(
            asker_id=asker_id,
            topic=topic,
            expert_id=None,
            reason="no suitable expert",
        )
    confidence = state.confidence(expert_id, topic) or 0.0
    return ConsultationRequest(
        asker_id=asker_id,
        topic=topic,
        expert_id=expert_id,
        reason=f"{expert_id} confidence {confidence:.2f} on {topic}",
    )


def consultation_samples(
    state: KnowledgeState,
    pool: SamplePool,
    asker_id: str,
    identity: StudentIdentity,
    exam_result: ExamResult,
    *,
    mission_seed: int = 0,
    samples_per_topic: int = 2,
) -> tuple[list[Sample], list[ConsultationRequest]]:
    """Pull pool samples for failed out-of-mission topics via expert routing."""
    requests: list[ConsultationRequest] = []
    collected: list[Sample] = []
    seen: set[str] = set()

    for failure in exam_result.failures():
        topic = failure.question.topic
        fit, _reason = domain_fit(topic, identity, seed=mission_seed)
        if fit > 0.0:
            continue
        request = request_consultation(state, asker_id, topic)
        requests.append(request)
        if request.expert_id is None:
            continue
        for sample in pool.collect(topics=[topic], limit=samples_per_topic):
            key = sample.input
            if key not in seen:
                collected.append(sample)
                seen.add(key)

    return collected, requests


def mediated_consultation_samples(
    state: KnowledgeState,
    grader: Grader,
    pool: SamplePool,
    asker_id: str,
    asker: Student,
    identity: StudentIdentity,
    exam_result: ExamResult,
    experts: dict[str, Student],
    *,
    mission_seed: int = 0,
    evidence_broker=None,
    show_me_on_reject: bool = False,
) -> tuple[list[Sample], list[MediatedConsultationResult]]:
    """Teacher-validated samples for failed out-of-mission topics."""
    from allm.teacher.mediated_consultation import MediatedConsultationResult, mediated_consultation

    collected: list[Sample] = []
    results: list[MediatedConsultationResult] = []
    seen: set[str] = set()

    for failure in exam_result.failures():
        topic = failure.question.topic
        fit, _reason = domain_fit(topic, identity, seed=mission_seed)
        if fit > 0.0:
            continue
        prompt = failure.question.prompt
        expected = failure.question.expected or ""
        if not expected:
            for sample in pool.collect(topics=[topic], limit=1):
                expected = sample.target or ""
                if not prompt:
                    prompt = sample.input
                break
        expert_request = request_consultation(state, asker_id, topic)
        expert = experts.get(expert_request.expert_id or "")
        if expert is None:
            continue
        result = mediated_consultation(
            state,
            grader,
            asker_id,
            expert,
            topic=topic,
            prompt=prompt,
            expected=expected,
            evidence_broker=evidence_broker,
            show_me_on_reject=show_me_on_reject,
        )
        results.append(result)
        if result.study_sample is not None and result.study_sample.input not in seen:
            collected.append(result.study_sample)
            seen.add(result.study_sample.input)

    return collected, results
