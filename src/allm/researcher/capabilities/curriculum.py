"""L4 — Curriculum building and recommendation targeting."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.topic_alignment import align_recommendation_topic
from allm.researcher.types import ResearchRecommendation
from allm.students.identity import StudentIdentity, domain_fit

logger = get_logger("researcher.curriculum")


def _student_topic_status(
    state,
    student_id: str,
    topic: str,
    *,
    mastery_threshold: float,
) -> str:
    """Return mastered, struggling, or unseen for one student-topic pair."""
    if state is None:
        return "unseen"
    confidence = state.confidence(student_id, topic)
    if confidence is None:
        return "unseen"
    if confidence >= mastery_threshold:
        return "mastered"
    if confidence > 0:
        return "struggling"
    return "unseen"


def _target_students(
    ctx: CapabilityContext,
    topic: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Who should learn vs skip this topic."""
    suggested: list[str] = []
    skip: list[str] = []
    threshold = ctx.config.mastery_threshold

    for student_id in ctx.student_ids:
        status = _student_topic_status(ctx.state, student_id, topic, mastery_threshold=threshold)
        identity = ctx.identities.get(student_id)
        if isinstance(identity, StudentIdentity):
            fit, _ = domain_fit(topic, identity, seed=0)
            if fit <= 0.0 and status == "unseen":
                continue
        if status == "mastered":
            skip.append(student_id)
        elif status in {"struggling", "unseen"}:
            if identity is None or isinstance(identity, StudentIdentity):
                if identity is None or domain_fit(topic, identity, seed=0)[0] > 0:
                    suggested.append(student_id)

    return tuple(suggested), tuple(skip)


class CurriculumTargetingCapability:
    """L4 — build recommendations with per-student targeting."""

    level = 4
    name = "curriculum.target"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        packages = pipeline.verified_packages or pipeline.packages
        reputation_by_provider = {
            d.provider_id: d.reputation_score for d in pipeline.discoveries
        }
        catalog = set(ctx.config.catalog_topics)
        recommendations: list[ResearchRecommendation] = []
        mission_by_topic = {
            topic: mission.id
            for mission in pipeline.active_missions
            for topic in mission.target_topics
        }

        for package in packages:
            reputation = reputation_by_provider.get(package.provider, 0.5)
            for concept in package.concepts:
                topic = align_recommendation_topic(
                    concept.name,
                    curriculum_topic=package.curriculum_topic,
                    catalog_topics=catalog,
                )
                priority = round(min(1.0, concept.confidence * reputation), 4)
                detail = concept.name if topic != concept.name else package.title
                suggested, skip = _target_students(ctx, topic)
                debate = len(package.conflicts) > 0
                hint = None
                if debate:
                    hint = f"Debate candidate: {concept.name} ({len(package.conflicts)} conflicts)"
                reason = f"Researcher: {detail} via {package.provider}"
                if package.distilled_visual_briefs:
                    reason += f" ({len(package.distilled_visual_briefs)} visual brief(s) for Teacher)"
                if package.student_visual_packages:
                    reason += f" [{len(package.student_visual_packages)} student visual(s) approved]"
                cross_report = getattr(pipeline, "cross_source_report", None)
                if cross_report is not None and getattr(cross_report, "aligned_count", 0) > 0:
                    if package.provider in {"kids-workshops", "keshe-books"}:
                        reason += (
                            f" [cross-source: {cross_report.aligned_count} aligned"
                            f" workshop↔book]"
                        )
                recommendations.append(
                    ResearchRecommendation(
                        topic=topic,
                        priority=priority,
                        reason=reason,
                        package_id=package.id,
                        provider=package.provider,
                        concept=concept.name if topic != concept.name else None,
                        suggested_students=suggested,
                        skip_students=skip,
                        debate_candidate=debate,
                        proposal_hint=hint,
                        mission_id=mission_by_topic.get(topic),
                    knowledge_tier=concept.knowledge_tier,
                    recommendation_kind="discovery",
                )
                )

        from allm.researcher.maintenance import build_maintenance_recommendations

        recommendations.extend(
            build_maintenance_recommendations(ctx.state, ctx.student_ids)
        )

        from allm.researcher.remediation import requests_to_recommendations

        if ctx.store is not None:
            from allm.kel.research_requests import KelResearchRequest

            kel_requests: list[KelResearchRequest] = []
            for key in ctx.store.keys("kel_research_requests"):
                record = ctx.store.get("kel_research_requests", key)
                if record is not None:
                    kel_requests.append(KelResearchRequest.model_validate(record.value))
            if kel_requests and not pipeline.curriculum_diagnostics:
                recommendations.extend(requests_to_recommendations(kel_requests))

        pipeline.recommendations = recommendations
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("curriculum.target: recommendations=%d", len(recommendations))
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(recommendations),
                duration_ms=round(elapsed, 2),
            ),
        )
