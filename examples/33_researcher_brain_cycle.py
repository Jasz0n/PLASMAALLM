"""Full Researcher brain cycle: observe → gap → missions → plan → discover.

    PYTHONPATH=src python3 examples/33_researcher_brain_cycle.py
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
    workdir = Path(tempfile.mkdtemp(prefix="allm-researcher-brain-"))
    store = SQLiteRecordStore(workdir / "brain.sqlite3")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC, description="Kids plasma", confidence=0.85))
    graph.add(Concept(name="fusion", prerequisites=("ions",), confidence=0.6))

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
        workshop_max_files=2,
        catalog_topics=(DEFAULT_TOPIC, "fastify-api", "ions"),
        graph=graph,
    )
    report = researcher.run_cycle()

    print("\n=== Researcher brain cycle (M6) ===")
    print(f"  curiosity signals: {len(report.curiosity_signals)}")
    for signal in report.curiosity_signals[:3]:
        print(f"    - [{signal.score:.2f}] {signal.question} ({signal.source})")

    print(f"\n  graph gaps: {len(report.graph_gaps)}")
    for gap in report.graph_gaps[:3]:
        print(f"    - missing {gap.missing_prerequisite!r} for {gap.parent!r} → {gap.child!r}")

    print(f"\n  active missions: {len(report.active_missions)}")
    for mission in report.active_missions[:3]:
        print(f"    - {mission.id}: {mission.goal} (priority={mission.priority:.2f})")

    if report.plan is not None:
        print(f"\n  plan goal: {report.plan.goal}")
        print(f"  plan priority: {report.plan.priority:.2f}")

    print(f"\n  packages: {len(report.packages)}")
    print(f"  recommendations: {len(report.recommendations)}")
    if report.recommendations:
        rec = report.recommendations[0]
        print(f"  first rec tier: {rec.knowledge_tier}, mission: {rec.mission_id}")

    print("\n=== Capability summary ===")
    for name, yield_count, notes in report.capability_summary:
        print(f"  {name}: yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
