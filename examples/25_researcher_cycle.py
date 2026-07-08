"""Researcher cycle demo — discover, package, recommend (offline).

    PYTHONPATH=src python3 examples/25_researcher_cycle.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.knowledge import Concept, KnowledgeGraph
from allm.planner import NeedPlanner, build_signals
from allm.planner.researcher_signals import merge_research_recommendations
from allm.researcher import ResearcherLayer
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-researcher-"))
    store = SQLiteRecordStore(workdir / "researcher.sqlite3")
    state = KnowledgeState(store)
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="kids-plasma", description="Kids plasma science"))

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
        workshop_max_files=2,
        catalog_topics=(DEFAULT_TOPIC, "fastify-api", "prisma-orm", "typescript-react"),
    )

    print("\n=== Researcher cycle ===")
    report = researcher.run_cycle()
    print(f"  providers evaluated: {report.providers_evaluated}")
    print(f"  packages built: {len(report.packages)}")
    print(f"  conflicts preserved: {report.conflicts_detected}")
    print(f"  recommendations: {len(report.recommendations)}")

    for package in report.packages:
        print(f"\n  Package: {package.title} ({package.id})")
        print(f"    concepts: {len(package.concepts)}  confidence: {package.confidence:.2f}")
        print(f"    provenance: {package.provenance}")

    print("\n=== Top recommendations (Researcher → Teacher) ===")
    for rec in report.recommendations[:8]:
        print(f"  [{rec.priority:.2f}] {rec.topic} — {rec.reason}")

    catalog = graph.to_catalog()
    boosted = merge_research_recommendations(dict(catalog), list(report.recommendations))
    signals = build_signals(state, "plasma-student", boosted)
    roadmap = NeedPlanner().plan("plasma-student", signals)

    print("\n=== Teacher roadmap (Researcher-boosted catalog) ===")
    for item in roadmap.items[:6]:
        print(f"  {item.rank}. {item.topic[:40]:<40} need={item.need:.3f}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
