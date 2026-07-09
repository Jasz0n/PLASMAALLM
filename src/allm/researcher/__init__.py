"""Researcher — distributed knowledge acquisition layer."""

from allm.researcher.types import (
    KnowledgePackage,
    PackageConcept,
    PackageConflict,
    ProviderReputation,
    ResearchRecommendation,
    ResearcherReport,
)

__all__ = [
    "KnowledgePackage",
    "PackageConcept",
    "PackageConflict",
    "ProviderReputation",
    "ResearchRecommendation",
    "ResearcherReport",
    "WorkshopProvider",
    "SoftwareFixtureProvider",
    "package_from_distillation",
    "package_from_repository",
    "package_from_workshop_dir",
    "package_from_samples_jsonl",
    "RepositoryProvider",
    "inject_package_concepts",
    "WorkshopLoop",
    "WorkshopReport",
    "WorkshopTick",
    "observer_source",
]


def __getattr__(name: str):
    """Lazy exports to avoid import cycles with KDP/knowledge/planner."""
    if name == "Provider":
        from allm.researcher.base import Provider

        return Provider
    if name == "Researcher":
        from allm.researcher.base import Researcher

        return Researcher
    if name == "providers":
        from allm.researcher.base import providers

        return providers
    if name == "ResearcherLayer":
        from allm.researcher.layer import ResearcherLayer

        return ResearcherLayer
    if name == "RecommendationQueue":
        from allm.researcher.queue import RecommendationQueue

        return RecommendationQueue
    if name in {
        "package_from_distillation",
        "package_from_repository",
        "package_from_workshop_dir",
        "package_from_samples_jsonl",
    }:
        from allm.researcher import packages

        return getattr(packages, name)
    if name in {"WorkshopProvider", "SoftwareFixtureProvider", "RepositoryProvider"}:
        from allm.researcher import providers as provider_mod

        return getattr(provider_mod, name)
    if name == "inject_package_concepts":
        from allm.researcher.graph_injection import inject_package_concepts

        return inject_package_concepts
    if name in {"WorkshopLoop", "WorkshopReport", "WorkshopTick", "observer_source"}:
        from allm.researcher import workshop_loop

        return getattr(workshop_loop, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
