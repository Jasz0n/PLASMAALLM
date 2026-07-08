"""Phase 5 demo: an evolving knowledge graph drives the curriculum.

Builds Plan.md's example chain (gravity -> general relativity -> black
holes -> quantum gravity), revises a belief with new evidence, shows the
preserved history, and hands the graph to the planner as a catalog.

Runs entirely offline:

    python examples/05_knowledge_graph.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.knowledge import Concept, Evidence, KnowledgeGraph
from allm.planner import NeedPlanner, TopicSignal
from allm.storage import SQLiteRecordStore

CHAIN = [
    Concept(
        name="gravity",
        description="masses attract each other",
        confidence=0.7,
        usefulness=0.9,
        curiosity=0.4,
        source="textbook",
        evidence=(Evidence(source="everyday observation", detail="things fall"),),
    ),
    Concept(
        name="general-relativity",
        description="gravity is curvature of spacetime",
        prerequisites=("gravity",),
        confidence=0.3,
        usefulness=0.7,
        curiosity=0.8,
        source="paper",
    ),
    Concept(
        name="black-holes",
        description="regions where curvature traps light",
        prerequisites=("general-relativity",),
        related=("gravity",),
        confidence=0.2,
        usefulness=0.5,
        curiosity=0.9,
        source="paper",
    ),
    Concept(
        name="quantum-gravity",
        description="open problem: unify GR with quantum mechanics",
        prerequisites=("general-relativity", "black-holes"),
        confidence=0.05,
        usefulness=0.4,
        curiosity=1.0,
        source="open questions list",
    ),
]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-kg-"))
    store = SQLiteRecordStore(workdir / "kg.sqlite3")
    graph = KnowledgeGraph(store)

    print("\n=== Building the concept chain ===")
    for concept in CHAIN:
        graph.add(concept)
        print(f"  + {concept.name} (confidence {concept.confidence:.2f})")

    print("\n=== Learning happens: revise with new evidence ===")
    graph.revise(
        "general-relativity",
        reason="passed the curved-spacetime exam",
        confidence=0.75,
        add_evidence=[Evidence(source="exam-0007", detail="scored 0.9")],
    )
    for version in graph.history("general-relativity"):
        print(f"  general-relativity v: confidence {version.confidence:.2f}, "
              f"evidence {[e.source for e in version.evidence]}")

    print("\n=== Graph structure ===")
    for name in graph.names():
        print(f"  {name}: prerequisites {list(graph.get(name).prerequisites)}, "
              f"dependents {graph.dependents_of(name)}")

    print("\n=== Graph as curriculum: planner roadmap ===")
    catalog = graph.to_catalog()
    signals = [
        TopicSignal(
            topic=concept.name,
            confidence=concept.confidence,
            importance=catalog[concept.name].importance,
            curiosity=catalog[concept.name].curiosity,
            dependencies=catalog[concept.name].dependencies,
            observations=len(graph.history(concept.name)) - 1,
        )
        for concept in graph.concepts()
    ]
    roadmap = NeedPlanner().plan("system", signals)
    for item in roadmap.items:
        blocked = f"  [blocked by {', '.join(item.blocked_by)}]" if item.blocked_by else ""
        print(f"  {item.rank}. {item.topic:<20} need {item.need:.3f}{blocked}")

    store.close()
    print(f"\nDone. Graph preserved at {workdir}/kg.sqlite3")


if __name__ == "__main__":
    main()
