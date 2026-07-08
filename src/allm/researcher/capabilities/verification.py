"""L3 — Verification against the knowledge graph."""

from __future__ import annotations

import time

from allm.core.logging import get_logger
from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
    VerificationReport,
)
from allm.researcher.knowledge_tier import classify_knowledge_tier
from allm.researcher.types import KnowledgePackage, PackageConcept

logger = get_logger("researcher.verification")


def _verify_package(package: KnowledgePackage, graph, reputation: float) -> tuple[KnowledgePackage, VerificationReport]:
    """Compare package concepts with graph; adjust confidence and tiers."""
    graph_names = set(graph.names()) if graph is not None else set()
    novel = existing = conflicting = 0
    proposal_hint = None
    has_conflict = len(package.conflicts) > 0

    tiered_concepts: list[PackageConcept] = []
    for concept in package.concepts:
        in_graph = concept.name in graph_names
        if in_graph:
            existing += 1
        else:
            novel += 1
        graph_conf = None
        if graph is not None and in_graph:
            node = graph.get(concept.name)
            if node is not None:
                graph_conf = node.confidence
        tier = classify_knowledge_tier(
            in_graph=in_graph,
            graph_confidence=graph_conf,
            has_conflict=has_conflict,
            package_confidence=concept.confidence,
        )
        tiered_concepts.append(concept.model_copy(update={"knowledge_tier": tier}))

    conflicting = len(package.conflicts)
    total = max(1, len(package.concepts))
    agreement = existing / total
    base = package.confidence
    adjusted = round(min(1.0, base * 0.5 + agreement * 0.3 + reputation * 0.2), 4)
    if conflicting > 0:
        adjusted = round(adjusted * 0.85, 4)
        proposal_hint = (
            f"Package {package.id} has {conflicting} preserved conflict(s) "
            f"on concepts from {package.provider}"
        )

    verified = package.model_copy(update={"confidence": adjusted, "concepts": tuple(tiered_concepts)})
    report = VerificationReport(
        package_id=package.id,
        novel_concepts=novel,
        existing_concepts=existing,
        conflicting_concepts=conflicting,
        adjusted_confidence=adjusted,
        proposal_hint=proposal_hint,
    )
    return verified, report


class GraphVerificationCapability:
    """L3 — verify packages against graph and provider reputation."""

    level = 3
    name = "verification.graph"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        reputation_by_provider = {
            d.provider_id: d.reputation_score for d in pipeline.discoveries
        }
        verified: list[KnowledgePackage] = []
        reports: list[VerificationReport] = []
        hints: list[str] = []

        for package in pipeline.packages:
            rep = reputation_by_provider.get(package.provider, 0.5)
            pkg, report = _verify_package(package, ctx.graph, rep)
            verified.append(pkg)
            reports.append(report)
            if report.proposal_hint:
                hints.append(report.proposal_hint)

        pipeline.verified_packages = verified
        pipeline.verification_reports = reports
        pipeline.proposal_hints = hints

        elapsed = (time.perf_counter() - started) * 1000
        logger.info(
            "verification.graph: verified=%d hints=%d",
            len(verified),
            len(hints),
        )
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(verified),
                duration_ms=round(elapsed, 2),
                notes=f"{len(hints)} proposal hints",
            ),
            artifacts={"reports": reports},
        )
