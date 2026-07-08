"""Built-in knowledge providers for Researcher v0."""

from __future__ import annotations

from pathlib import Path

from allm.researcher.base import providers
from allm.researcher.types import ProviderKind, ProviderReputation


class WorkshopProvider:
    """Kids workshop transcripts (cleaned MK prose)."""

    provider_id = "kids-workshops"
    kind: ProviderKind = "workshop"

    def __init__(self, corpus_dir: Path | str) -> None:
        self._dir = Path(corpus_dir)

    def discover(self) -> list[Path]:
        if not self._dir.is_dir():
            return []
        return sorted(self._dir.glob("*.txt"))

    def reputation(self) -> ProviderReputation:
        count = len(self.discover())
        return ProviderReputation(
            provider_id=self.provider_id,
            kind=self.kind,
            accuracy=0.85,
            freshness=min(1.0, count / 22.0),
            packages_submitted=count,
            packages_accepted=count,
        )


@providers.register("workshop")
class RegisteredWorkshopProvider(WorkshopProvider):
    """Registry entry for workshop provider."""


class SoftwareFixtureProvider:
    """AI-friendly dev curriculum fixture (Fastify, TypeScript, etc.)."""

    provider_id = "software-fixture"
    kind: ProviderKind = "software"

    def __init__(self, samples_path: Path | str) -> None:
        self._path = Path(samples_path)

    def discover(self) -> list[Path]:
        return [self._path] if self._path.is_file() else []

    def reputation(self) -> ProviderReputation:
        return ProviderReputation(
            provider_id=self.provider_id,
            kind=self.kind,
            accuracy=0.9,
            freshness=0.8,
            packages_submitted=1,
            packages_accepted=1,
        )


@providers.register("software")
class RegisteredSoftwareProvider(SoftwareFixtureProvider):
    """Registry entry for software fixture provider."""


class RepositoryProvider:
    """A real software repository as a knowledge source (Roadmap M49).

    Discovers the knowledge-bearing files of a working codebase: markdown
    documentation, project manifests, and Python sources (whose module
    docstrings carry design intent). Replaces the software fixture with
    ground truth — the repo is what the project *actually is*.
    """

    provider_id = "repository"
    kind: ProviderKind = "repository"

    MANIFESTS = ("pyproject.toml", "package.json", "Cargo.toml", "go.mod")

    def __init__(self, repo_dir: Path | str, *, max_files: int | None = 48) -> None:
        self._dir = Path(repo_dir)
        self._max_files = max_files
        self.provider_id = f"repo-{self._dir.name}"

    def discover(self) -> list[Path]:
        if not self._dir.is_dir():
            return []
        paths: list[Path] = []
        paths.extend(sorted(self._dir.glob("*.md")))
        docs = self._dir / "docs"
        if docs.is_dir():
            paths.extend(sorted(docs.rglob("*.md")))
        for manifest in self.MANIFESTS:
            candidate = self._dir / manifest
            if candidate.is_file():
                paths.append(candidate)
        src = self._dir / "src"
        code_root = src if src.is_dir() else self._dir
        paths.extend(
            p
            for p in sorted(code_root.rglob("*.py"))
            if "__pycache__" not in p.parts and ".venv" not in p.parts
        )
        return paths[: self._max_files] if self._max_files else paths

    def reputation(self) -> ProviderReputation:
        count = len(self.discover())
        return ProviderReputation(
            provider_id=self.provider_id,
            kind=self.kind,
            accuracy=0.9,  # ground truth: the repo is the project
            freshness=1.0 if count else 0.0,
            packages_submitted=count,
            packages_accepted=count,
        )


@providers.register("repository")
class RegisteredRepositoryProvider(RepositoryProvider):
    """Registry entry for the real-repository provider."""


class BookProvider:
    """Keshe foundation books (PDF) for plasma curriculum enrichment."""

    provider_id = "keshe-books"
    kind: ProviderKind = "book"

    def __init__(self, books_dir: Path | str) -> None:
        self._dir = Path(books_dir)

    def discover(self) -> list[Path]:
        if not self._dir.is_dir():
            return []
        return sorted(self._dir.glob("*.pdf"))

    def reputation(self) -> ProviderReputation:
        count = len(self.discover())
        return ProviderReputation(
            provider_id=self.provider_id,
            kind=self.kind,
            accuracy=0.88,
            freshness=min(1.0, count / 3.0),
            packages_submitted=count,
            packages_accepted=count,
        )


@providers.register("book")
class RegisteredBookProvider(BookProvider):
    """Registry entry for Keshe book provider."""
