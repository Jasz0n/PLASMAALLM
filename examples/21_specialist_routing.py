"""Specialist students: mission-aware planning and ingest routing (offline).

Shows how incoming concepts route to plasma vs software specialists,
how mission weights change the roadmap, and how expert lookup picks
the best student for a topic.

    PYTHONPATH=src python3 examples/21_specialist_routing.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.knowledge import Concept, KnowledgeGraph
from allm.planner import IngestRouter, NeedPlanner, build_signals
from allm.students import ScriptedStudent, load_identities_dir
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, best_expert, rank_experts

ROOT = Path(__file__).resolve().parents[1]
STUDENTS = ROOT / "configs/students"


def main() -> None:
    setup_logging("INFO")
    identities = load_identities_dir(STUDENTS)
    plasma = identities["plasma-student"]
    software = identities["software-student"]

    workdir = Path(tempfile.mkdtemp(prefix="allm-specialists-"))
    store = SQLiteRecordStore(workdir / "state.sqlite3")
    graph = KnowledgeGraph(store)
    concepts = [
        Concept(name="kids-plasma", description="Kids plasma workshops", usefulness=1.0),
        Concept(name="magnetic-fields", description="Magnetic interactions", usefulness=0.9),
        Concept(name="fastify-api", description="Fastify HTTP services", usefulness=0.9),
        Concept(name="typescript-react", description="React with TypeScript", usefulness=0.85),
        Concept(name="medieval-history", description="Medieval Europe", usefulness=0.2),
    ]
    for concept in concepts:
        graph.add(concept)

    catalog = graph.to_catalog()
    router = IngestRouter(identities.values(), seed=42)

    print("\n=== Ingest routing (concept -> students) ===")
    for concept in graph.names():
        students = router.students_for(concept)
        print(f"  {concept:<22} -> {', '.join(students) or '(none)'}")

    state = KnowledgeState(store)
    samples = [
        Sample(id="p1", input="Plasma?", target="fields", metadata={"topic": "kids-plasma"}),
        Sample(id="s1", input="Fastify?", target="framework", metadata={"topic": "fastify-api"}),
    ]
    teacher = Teacher(state, DatasetExamGenerator(samples), ExactMatchGrader("contains"))
    teacher.evaluate(
        ScriptedStudent("plasma-student", "kids-plasma", knowledge={"Plasma?": "fields"}),
        teacher.create_exam(num_questions=1, seed=1),
    )
    teacher.evaluate(
        ScriptedStudent("software-student", "fastify-api", knowledge={"Fastify?": "framework"}),
        teacher.create_exam(num_questions=1, seed=2),
    )

    print("\n=== Expert lookup ===")
    for topic in ("kids-plasma", "fastify-api"):
        expert = best_expert(state, topic)
        ranking = rank_experts(state, topic)
        scores = ", ".join(f"{sid}={score:.2f}" for sid, score in ranking.rankings)
        print(f"  {topic}: best={expert}  ({scores})")

    print("\n=== Roadmap: plasma student (mission-weighted) ===")
    plasma_signals = build_signals(state, plasma.student_id, catalog, identity=plasma)
    for item in NeedPlanner().plan(plasma.student_id, plasma_signals).items[:5]:
        print(f"  {item.rank}. {item.topic:<22} need={item.need:.3f} importance={item.importance:.2f}")

    print("\n=== Roadmap: software student (mission-weighted) ===")
    software_signals = build_signals(state, software.student_id, catalog, identity=software)
    for item in NeedPlanner().plan(software.student_id, software_signals).items[:5]:
        print(f"  {item.rank}. {item.topic:<22} need={item.need:.3f} importance={item.importance:.2f}")

    store.close()
    print(f"\nDone. State at {workdir}")


if __name__ == "__main__":
    main()
