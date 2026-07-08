"""Phase 10 demo: the full continuous learning loop, offline.

Two model-backed students run Plan.md's cycle for three iterations:
measure -> plan -> collect -> learn -> debate -> test -> compress ->
update memory. Watch scores rise, goals shift, the graph abstract, and
the evaluation metrics summarise it all.

    python examples/06_continuous_learning_loop.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.compression import CompressionEngine
from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.debate import DebateEngine
from allm.evaluation import evaluate_student
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import LearningLoop, LoopConfig
from allm.memory import EpisodicMemory
from allm.models import EchoModel, ModelSpec
from allm.planner import NeedPlanner
from allm.students import FailureLog, ModelStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.tracking import LocalTracker
from allm.trainer import InContextTrainer

FACTS = {
    "2+2?": ("4", "math"),
    "3*3?": ("9", "math"),
    "10/2?": ("5", "math"),
    "Capital of France?": ("Paris", "geography"),
    "Capital of Japan?": ("Tokyo", "geography"),
    "Capital of Egypt?": ("Cairo", "geography"),
}


def main() -> None:
    setup_logging("WARNING")  # keep the demo output readable
    workdir = Path(tempfile.mkdtemp(prefix="allm-loop-"))
    store = SQLiteRecordStore(workdir / "allm.sqlite3")

    samples = [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, (a, topic)) in enumerate(FACTS.items())
    ]
    teacher = Teacher(
        KnowledgeState(store),
        DatasetExamGenerator(samples),
        ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=0.5),
    )

    graph = KnowledgeGraph(store)
    graph.add(Concept(name="symbol-manipulation", description="working with symbols"))
    graph.add(Concept(name="math", prerequisites=("symbol-manipulation",), usefulness=0.9))
    graph.add(Concept(name="geography", prerequisites=("symbol-manipulation",), usefulness=0.6))

    pool = SamplePool()
    pool.ingest(samples)
    memory = EpisodicMemory(store)
    run = LocalTracker(workdir / "runs").start_run("continuous-learning")

    students = [
        ModelStudent(sid, "general", EchoModel(ModelSpec(name=sid, provider="echo", model_id="none")))
        for sid in ("alpha", "beta")
    ]

    loop = LearningLoop(
        teacher=teacher,
        students=students,
        planner=NeedPlanner(),
        trainer=InContextTrainer(),
        pool=pool,
        memory=memory,
        failure_log=FailureLog(store),
        graph=graph,
        compression=CompressionEngine(graph),
        debate=DebateEngine(grader=ExactMatchGrader()),
        run=run,
        config=LoopConfig(iterations=3, questions_per_exam=6, seed=11),
    )

    print("\n=== Running the loop (3 iterations) ===")
    for report in loop.run():
        print(f"\niteration {report.iteration}:")
        for s in report.students:
            print(f"  {s.student_id}: {s.score_before:.2f} -> {s.score_after:.2f}"
                  f"  goals={list(s.goals)}  studied={s.samples_studied}")
        if report.debate_disagreement is not None:
            print(f"  debate disagreement: {report.debate_disagreement:.2f}")
        if report.compression_applied:
            print(f"  compression: {report.compression_applied} principle(s) formed")

    print("\n=== Knowledge graph after compression ===")
    for name in graph.names():
        concept = graph.get(name)
        print(f"  {name} [{concept.status}] related={list(concept.related)}")

    print("\n=== Plan.md success metrics ===")
    for student in students:
        ev = evaluate_student(teacher.state, memory, student.student_id)
        print(f"  {ev.student_id}: learning_speed={ev.learning_speed:+.2f} "
              f"mastery={ev.mastery:.2f} self_correction={ev.self_correction_rate}")
        for topic, delta in sorted(ev.improvement_per_topic.items()):
            print(f"    {topic}: {delta:+.2f}")

    print(f"\n=== Memory ===\n  {len(memory.recall())} episodes recorded")
    failures_recalled = memory.search("missed capital")
    if failures_recalled:
        print(f"  e.g. remembered failure: {failures_recalled[0].summary}")

    store.close()
    print(f"\nDone. Everything preserved under {workdir}")


if __name__ == "__main__":
    main()
