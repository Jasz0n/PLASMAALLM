"""Researcher ecosystem metrics feeding KEL diagnosis (offline).

    PYTHONPATH=src python3 examples/28_researcher_kel_metrics.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import Concept, KnowledgeGraph
from allm.researcher import ResearcherLayer
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-researcher-kel-metrics-"))
    store = SQLiteRecordStore(workdir / "metrics.sqlite3")
    state = KnowledgeState(store)
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="kids-plasma", description="Kids plasma science"))
    kel = KnowledgeEvaluationLayer(graph, store, state)

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
        workshop_max_files=2,
    )
    report = researcher.run_cycle()
    ecosystem = researcher.ecosystem_metrics(graph, state)

    print("\n=== Researcher ecosystem metrics ===")
    print(f"  missing_knowledge:     {ecosystem.missing_knowledge:.2f}")
    print(f"  research_saturation:   {ecosystem.research_saturation:.2f}")
    print(f"  high_conflict_areas:   {ecosystem.high_conflict_areas:.2f}")
    print(f"  emerging_topics:       {ecosystem.emerging_topics}")
    print(f"  source_reliability:    {ecosystem.source_reliability:.2f}")
    print(f"  knowledge_growth_rate: {ecosystem.knowledge_growth_rate:.2f}")
    print(f"  packages: {ecosystem.package_count}  conflicts: {ecosystem.conflict_count}")

    kel.evaluate(ecosystem=ecosystem)
    findings = kel.diagnose()
    print("\n=== KEL diagnosis (Researcher-informed) ===")
    for finding in findings:
        print(f"  [{finding.mode}] {finding.detail}")
    if not findings:
        print("  (no failure modes at current thresholds)")

    print(f"\nResearcher cycle: {len(report.recommendations)} recommendations")
    store.close()
    print(f"Done. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
