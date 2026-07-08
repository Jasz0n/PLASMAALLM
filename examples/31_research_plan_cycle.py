"""Research plan + capability pipeline demo (offline).

    PYTHONPATH=src python3 examples/31_research_plan_cycle.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.knowledge import Concept, KnowledgeGraph
from allm.researcher import ResearcherLayer
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-research-plan-"))
    store = SQLiteRecordStore(workdir / "plan.sqlite3")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC, description="Kids plasma"))

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
        workshop_max_files=2,
        catalog_topics=(DEFAULT_TOPIC, "fastify-api", "prisma-orm"),
        graph=graph,
    )
    report = researcher.run_cycle()

    print("\n=== Research plan + capability cycle ===")
    if report.plan is not None:
        print(f"  goal: {report.plan.goal}")
        print(f"  priority: {report.plan.priority:.2f}")
        print(f"  steps: {len(report.plan.steps)}")
        for step in report.plan.steps[:6]:
            print(f"    - {step.capability}: {step.detail}")

    print(f"\n  packages: {len(report.packages)}")
    print(f"  recommendations: {len(report.recommendations)}")
    print(f"  conflicts preserved: {report.conflicts_detected}")
    print(f"  proposal hints: {len(report.proposal_hints)}")

    print("\n=== Capability summary ===")
    for name, yield_count, notes in report.capability_summary:
        print(f"  {name}: yield={yield_count} ({notes})")

    if report.packages:
        pkg = report.packages[0]
        print(f"\n  first package confidence (post-verify): {pkg.confidence:.2f}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
