"""Phase 4 demo: from measured state to an ordered learning roadmap.

A student is examined on two topics; the planner merges that state with
a curriculum catalog (importance, curiosity, prerequisites) and produces
a prioritised roadmap where prerequisites come before the exciting
advanced topics that depend on them.

Runs entirely offline:

    python examples/04_learning_roadmap.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.planner import NeedPlanner, build_signals, load_catalog
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher

CATALOG = Path(__file__).resolve().parent.parent / "configs/curriculum/example.yaml"

FACTS = {
    "7+5?": ("12", "arithmetic"),
    "9-3?": ("6", "arithmetic"),
    "Capital of France?": ("Paris", "geography"),
    "Capital of Japan?": ("Tokyo", "geography"),
}


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-planner-"))
    store = SQLiteRecordStore(workdir / "state.sqlite3")
    state = KnowledgeState(store)

    samples = [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, (a, topic)) in enumerate(FACTS.items())
    ]
    teacher = Teacher(state, DatasetExamGenerator(samples), ExactMatchGrader())

    # The student is decent at arithmetic, clueless at geography.
    student = ScriptedStudent(
        "student-1", "arithmetic", knowledge={"7+5?": "12", "9-3?": "6"}
    )
    teacher.evaluate(student, teacher.create_exam(num_questions=4, seed=3))

    print("\n=== Measured state ===")
    for topic in state.topics("student-1"):
        print(f"  {topic}: confidence {state.confidence('student-1', topic):.2f}")

    print("\n=== Roadmap (Need = Weakness x Importance x Curiosity x Novelty) ===")
    catalog = load_catalog(CATALOG)
    signals = build_signals(state, "student-1", catalog)
    roadmap = NeedPlanner().plan("student-1", signals)
    for item in roadmap.items:
        blocked = f"  [blocked by {', '.join(item.blocked_by)}]" if item.blocked_by else ""
        print(f"  {item.rank}. {item.topic:<20} need {item.need:.3f}{blocked}")

    print("\n=== Top goals handed to the teacher ===")
    goals = roadmap.to_goals(max_goals=3)
    teacher.state.record_goals("student-1", goals)
    for goal in goals:
        print(f"  study {goal.topic!r} (priority {goal.priority:.3f}) — {goal.reason}")

    store.close()
    print(f"\nDone. State preserved at {workdir}/state.sqlite3")


if __name__ == "__main__":
    main()
