"""M49 slice 1: the Researcher reads a REAL repository — this one.

The software fixture is replaced by ground truth: the Researcher
discovers ALLM's own codebase (markdown docs, manifests, module
docstrings), distills it through KDP into a Knowledge Package with full
provenance, and enqueues recommendations — the first step toward the
open-source apprentice that studies a project before contributing to it.

    PYTHONPATH=src python3 examples/77_repository_researcher.py
    ALLM_REPO_DIR=/path/to/any/repo PYTHONPATH=src python3 examples/77_repository_researcher.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.knowledge import KnowledgeGraph
from allm.planner import NeedPlanner, build_signals
from allm.planner.researcher_signals import merge_research_recommendations
from allm.researcher import ResearcherLayer, inject_package_concepts
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("WARNING")
    repo_dir = Path(os.environ.get("ALLM_REPO_DIR", ROOT))
    workdir = Path(tempfile.mkdtemp(prefix="allm-repo-researcher-"))
    store = SQLiteRecordStore(workdir / "researcher.sqlite3")
    graph = KnowledgeGraph(store)

    researcher = ResearcherLayer(
        store,
        repository_dir=repo_dir,
        repository_max_files=int(os.environ.get("ALLM_REPO_MAX_FILES", "24")),
        enabled_capabilities=(
            "planning.research",
            "discovery.repository",
            "understanding.package",
            "verification.graph",
            "curriculum.target",
        ),
        graph=graph,
    )

    print(f"=== Researcher studies a real repository: {repo_dir.name} ===")
    report = researcher.run_cycle()
    print(f"providers evaluated: {report.providers_evaluated}")
    print(f"packages built: {len(report.packages)}")
    print(f"conflicts preserved: {report.conflicts_detected}")

    for package in report.packages:
        print(f"\nPackage: {package.title} ({package.id})")
        print(f"  curriculum topic: {package.curriculum_topic}")
        print(f"  confidence: {package.confidence:.2f}")
        print(f"  provenance: {package.provenance}")
        print(f"  concepts ({len(package.concepts)}), first 10:")
        for concept in package.concepts[:10]:
            description = concept.description[:70].replace("\n", " ")
            print(f"    - {concept.name}: {description}")
        if package.procedures:
            print(f"  procedures captured: {len(package.procedures)}")

    print("\n=== Recommendations (Researcher → Teacher; never taught directly) ===")
    for rec in report.recommendations[:8]:
        print(f"  [{rec.priority:.2f}] {rec.topic} — {rec.reason[:80]}")
    if not report.recommendations:
        print("  (none this cycle)")

    print("\n=== Study roadmap from the graph alone (no raw text) ===")
    for package in report.packages:
        counts = inject_package_concepts(graph, package)
        print(f"graph injection: {counts['added']} added, {counts['revised']} revised")
    catalog = merge_research_recommendations(
        graph.to_catalog(), list(report.recommendations)
    )
    signals = build_signals(KnowledgeState(store), "apprentice", catalog)
    roadmap = NeedPlanner().plan("apprentice", signals)
    for item in roadmap.items[:10]:
        print(f"  [{item.need:.2f}] {item.topic} — {item.reason[:70]}")

    store.close()
    print(f"\nworkdir: {workdir}")


if __name__ == "__main__":
    main()
