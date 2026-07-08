"""Real-repository provider (Roadmap M49 slice 1), offline."""

from pathlib import Path

import pytest

from allm.researcher.capabilities.base import (
    CapabilityContext,
    PipelineState,
    ResearcherPipelineConfig,
)
from allm.researcher.capabilities.discovery import RepositoryDiscoveryCapability
from allm.researcher.capabilities.understanding import PackageUnderstandingCapability
from allm.researcher.packages import module_docstring, package_from_repository
from allm.researcher.providers import RepositoryProvider
from allm.storage import SQLiteRecordStore

README = """# widgetlib

Widgetlib is a Python library for building configurable widgets.
A widget is a reusable interface component with a declared configuration.
The registry is the central catalog that maps widget names to factories.
"""

MODULE = '''"""The widget registry: maps widget names to their factories.

The registry is deliberately append-only: a registered widget factory
is never replaced, so behaviour cannot change silently.
"""

WIDGETS = {}
'''


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "widgetlib"
    (root / "src" / "widgetlib").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "README.md").write_text(README)
    (root / "docs" / "design.md").write_text(
        "## Design\nEvery widget factory validates its configuration before construction.\n"
    )
    (root / "pyproject.toml").write_text('[project]\nname = "widgetlib"\n')
    (root / "src" / "widgetlib" / "registry.py").write_text(MODULE)
    (root / "src" / "widgetlib" / "broken.py").write_text("def :::\n")  # unparseable
    return root


def test_provider_discovers_knowledge_bearing_files(repo: Path) -> None:
    provider = RepositoryProvider(repo)
    paths = [p.name for p in provider.discover()]
    assert "README.md" in paths
    assert "design.md" in paths
    assert "pyproject.toml" in paths
    assert "registry.py" in paths
    assert provider.provider_id == "repo-widgetlib"
    assert provider.reputation().score > 0


def test_module_docstring_extraction(repo: Path) -> None:
    doc = module_docstring(repo / "src" / "widgetlib" / "registry.py")
    assert doc is not None and "append-only" in doc
    assert module_docstring(repo / "src" / "widgetlib" / "broken.py") is None


def test_package_from_repository_carries_provenance(repo: Path) -> None:
    package = package_from_repository(repo, provider="repo-widgetlib")
    assert package.provenance == str(repo)
    assert package.curriculum_topic == "software-widgetlib"
    assert package.concepts  # KDP found concepts in the docs
    names = " ".join(c.name.lower() for c in package.concepts)
    assert "widget" in names or "registry" in names


def test_capabilities_discover_and_package(tmp_path: Path, repo: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "researcher.sqlite3")
    try:
        ctx = CapabilityContext(
            store=store,
            config=ResearcherPipelineConfig(repository_dir=repo),
        )
        pipeline = PipelineState()
        result = RepositoryDiscoveryCapability().run(ctx, pipeline)
        assert result.metrics.yield_count > 0
        assert pipeline.discoveries[0].kind == "repository"

        PackageUnderstandingCapability().run(ctx, pipeline)
        (package,) = pipeline.packages
        assert package.provider == "repo-widgetlib"
        assert package.concepts
    finally:
        store.close()


def test_discovery_skips_when_unconfigured(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "researcher.sqlite3")
    try:
        ctx = CapabilityContext(store=store, config=ResearcherPipelineConfig())
        result = RepositoryDiscoveryCapability().run(ctx, PipelineState())
        assert result.metrics.yield_count == 0
        assert result.metrics.notes == "no repository configured"
    finally:
        store.close()


def test_injected_package_becomes_plannable_curriculum(tmp_path: Path, repo: Path) -> None:
    from allm.knowledge import KnowledgeGraph
    from allm.planner import NeedPlanner, build_signals
    from allm.researcher.graph_injection import (
        INJECTION_CONFIDENCE_CAP,
        inject_package_concepts,
    )
    from allm.teacher import KnowledgeState

    package = package_from_repository(repo, provider="repo-widgetlib")
    store = SQLiteRecordStore(tmp_path / "graph.sqlite3")
    try:
        graph = KnowledgeGraph(store)
        counts = inject_package_concepts(graph, package)
        assert counts["added"] == len(package.concepts)

        # documents propose, evidence disposes: injection never exceeds the cap
        for concept in graph.concepts():
            assert concept.confidence <= INJECTION_CONFIDENCE_CAP
            assert concept.evidence and concept.evidence[0].source == package.id

        # planner builds a study roadmap from the graph alone — no raw text
        signals = build_signals(KnowledgeState(store), "apprentice", graph.to_catalog())
        roadmap = NeedPlanner().plan("apprentice", signals)
        assert len(roadmap.items) >= len(package.concepts)
        assert all(item.need > 0 for item in roadmap.items)

        # re-injection is additive, never duplicating
        again = inject_package_concepts(graph, package)
        assert again["added"] == 0 and again["revised"] == len(package.concepts)
    finally:
        store.close()
