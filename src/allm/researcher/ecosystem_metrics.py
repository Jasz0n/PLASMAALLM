"""Researcher ecosystem metrics for KEL (ResearcherPlan §Interaction with KEL)."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.capabilities.ecosystem import StudentTopicMatrix, build_student_topic_matrix
from allm.knowledge.graph import KnowledgeGraph
from allm.researcher.types import KnowledgePackage, ResearchRecommendation
from allm.storage.base import RecordStore
from allm.teacher.state import KnowledgeState

SNAPSHOT_NAMESPACE = "researcher_ecosystem"
SNAPSHOT_KEY = "last"


class ResearcherEcosystemMetrics(BaseModel):
    """Signals from the Researcher layer for KEL and curriculum steering."""

    model_config = ConfigDict(frozen=True)

    missing_knowledge: float = Field(default=0.0, ge=0.0, le=1.0)
    research_saturation: float = Field(default=0.0, ge=0.0, le=1.0)
    high_conflict_areas: float = Field(default=0.0, ge=0.0, le=1.0)
    emerging_topics: int = Field(default=0, ge=0)
    outdated_concepts: float = Field(default=0.0, ge=0.0, le=1.0)
    source_reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    knowledge_growth_rate: float = Field(default=0.0)
    package_count: int = Field(default=0, ge=0)
    recommendation_count: int = Field(default=0, ge=0)
    conflict_count: int = Field(default=0, ge=0)
    student_topic_matrix: StudentTopicMatrix | None = None


def compute_ecosystem_metrics(
    graph: KnowledgeGraph,
    state: KnowledgeState,
    recommendations: list[ResearchRecommendation],
    packages: list[KnowledgePackage],
    *,
    store: RecordStore | None = None,
    mastery_threshold: float = 0.75,
    emerging_priority: float = 0.5,
    student_ids: tuple[str, ...] = (),
) -> ResearcherEcosystemMetrics:
    """Derive ecosystem metrics from graph, teacher state, and Researcher output."""
    graph_topics = set(graph.names())
    rec_topics = {rec.topic for rec in recommendations}
    missing = rec_topics - graph_topics
    missing_knowledge = round(len(missing) / max(1, len(rec_topics)), 4)

    mastered = _mastered_topics(state, mastery_threshold)
    saturated = rec_topics & mastered
    research_saturation = round(len(saturated) / max(1, len(rec_topics)), 4)

    conflict_count = sum(len(package.conflicts) for package in packages)
    high_conflict_areas = round(
        min(1.0, conflict_count / max(1, len(packages) * 2)),
        4,
    )

    emerging_topics = sum(
        1
        for rec in recommendations
        if rec.priority >= emerging_priority and rec.topic not in mastered
    )

    confidences = [package.confidence for package in packages]
    source_reliability = round(
        sum(confidences) / len(confidences) if confidences else 0.5,
        4,
    )

    concept_count = sum(len(package.concepts) for package in packages)
    growth_rate = 0.0
    if store is not None:
        previous = _load_snapshot(store)
        if previous is not None and previous > 0:
            growth_rate = round((concept_count - previous) / previous, 4)
        _save_snapshot(store, concept_count)

    matrix = None
    if student_ids:
        matrix = build_student_topic_matrix(
            state,
            student_ids,
            rec_topics | graph_topics,
            mastery_threshold=mastery_threshold,
        )
    outdated = 0.0
    if matrix is not None and research_saturation > 0.5:
        outdated = round(research_saturation * 0.5, 4)

    return ResearcherEcosystemMetrics(
        missing_knowledge=missing_knowledge,
        research_saturation=research_saturation,
        high_conflict_areas=high_conflict_areas,
        emerging_topics=emerging_topics,
        outdated_concepts=outdated,
        source_reliability=source_reliability,
        knowledge_growth_rate=growth_rate,
        package_count=len(packages),
        recommendation_count=len(recommendations),
        conflict_count=conflict_count,
        student_topic_matrix=matrix,
    )


def _mastered_topics(state: KnowledgeState, threshold: float) -> set[str]:
    mastered: set[str] = set()
    for student_id in state.students():
        for topic in state.topics(student_id):
            confidence = state.confidence(student_id, topic)
            if confidence is not None and confidence >= threshold:
                mastered.add(topic)
    return mastered


def _load_snapshot(store: RecordStore) -> int | None:
    record = store.get(SNAPSHOT_NAMESPACE, SNAPSHOT_KEY)
    if record is None:
        return None
    return int(record.value.get("concept_count", 0))


def _save_snapshot(store: RecordStore, concept_count: int) -> None:
    store.put(
        SNAPSHOT_NAMESPACE,
        SNAPSHOT_KEY,
        {"concept_count": concept_count},
        reason="researcher ecosystem snapshot",
    )
