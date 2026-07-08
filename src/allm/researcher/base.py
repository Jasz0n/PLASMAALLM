"""Researcher and Provider protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from allm.core.registry import Registry
from allm.researcher.types import KnowledgePackage, ProviderReputation


@runtime_checkable
class Provider(Protocol):
    """External knowledge source the Researcher can discover."""

    provider_id: str
    kind: str

    def discover(self) -> list[Path]:
        """Return paths to new or updated source documents."""

    def reputation(self) -> ProviderReputation:
        """Current dynamic trust for this provider."""


@runtime_checkable
class Researcher(Protocol):
    """Distributed knowledge acquisition — recommends, never teaches."""

    def run_cycle(self) -> "ResearcherReport":
        """Discover, evaluate, package, and enqueue recommendations."""


providers: Registry[type] = Registry("researcher_provider")

from allm.researcher.types import ResearcherReport  # noqa: E402
