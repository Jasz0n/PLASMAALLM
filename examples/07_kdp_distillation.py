"""KDP demo: noisy transcripts -> knowledge units -> graph -> roadmap.

Two overlapping workshop transcripts (fillers, stutters, a
contradiction included) are distilled into atomic knowledge units,
injected into the Phase 5 knowledge graph, and immediately ranked by
the Phase 4 planner — the KDP.md success criterion "planner can rank
learning priorities without raw text", end to end.

    python examples/07_kdp_distillation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.knowledge import KnowledgeGraph
from allm.planner import NeedPlanner, TopicSignal
from allm.storage import SQLiteRecordStore

WORKSHOP_1 = """Um, so today we talk about attention. Self-attention is a
mechanism that relates every token to every other token. It computes
weights between all pairs, you know, all 100% of them.

People often think attention reads the the sentence word by word, but
that is not how it works.

Dropout is a regularisation technique that removes random units during
training.

How does self attention scale with long sequences?
"""

WORKSHOP_2 = """The self attention mechanism is a weighted lookup across all
token pairs.

To compute it: first project the inputs into queries, keys and values,
then compare queries with keys, finally weight the values by the scores.

Dropout is the compression of gradients so that training synchronises
faster across machines.
"""


def main() -> None:
    setup_logging("WARNING")
    workdir = Path(tempfile.mkdtemp(prefix="allm-kdp-"))
    store = SQLiteRecordStore(workdir / "allm.sqlite3")

    print("=== Stages 1-8: distillation ===")
    documents = DocumentStore(store)
    documents.ingest_text("workshop-1.md", WORKSHOP_1, context="transformers")
    documents.ingest_text("workshop-2.md", WORKSHOP_2, context="transformers")
    result = KDPipeline().distill(documents)
    print(f"{result.documents} docs -> {result.segments} segments -> "
          f"{len(result.units)} knowledge units, {len(result.conflicts)} conflict(s)\n")
    for unit in result.units:
        print(f"  {unit.id}")
        print(f"    [{unit.type}] {unit.normalized_concept}  "
              f"confidence={unit.confidence:.2f}  sources={list(unit.sources)}")

    print("\n=== Stage 7 output: preserved contradiction ===")
    for conflict in result.conflicts:
        print(f"  {conflict.concept}:")
        print(f"    A: {conflict.interpretation_a[:70]}...")
        print(f"    B: {conflict.interpretation_b[:70]}...")

    print("\n=== Stage 9: graph injection (append-only, provenance kept) ===")
    graph = KnowledgeGraph(store)
    report = GraphInjector(graph, store).inject(result)
    print(f"  {report['added']} concepts added, {report['conflicts']} conflict(s) stored")
    for name in graph.names():
        concept = graph.get(name)
        print(f"  {name}: confidence={concept.confidence:.2f}, "
              f"{len(concept.evidence)} evidence item(s)")

    print("\n=== Success criterion: planner ranks without raw text ===")
    signals = [
        TopicSignal(
            topic=c.name,
            confidence=c.confidence,
            importance=c.usefulness,
            curiosity=c.curiosity,
            dependencies=c.prerequisites,
        )
        for c in graph.concepts()
    ]
    for item in NeedPlanner().plan("system", signals).items:
        print(f"  {item.rank}. {item.topic:<20} need {item.need:.3f}")

    store.close()
    print(f"\nDone. Everything preserved under {workdir}")


if __name__ == "__main__":
    main()
