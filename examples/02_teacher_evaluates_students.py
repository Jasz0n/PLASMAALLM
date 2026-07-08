"""Phase 2 demo: a teacher examines two students, finds weaknesses,
assigns goals, and measures progress after they study.

Runs entirely offline (scripted students, temp storage):

    python examples/02_teacher_evaluates_students.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig

MATH = {"2+2?": "4", "3*3?": "9", "10/2?": "5"}
GEOGRAPHY = {
    "Capital of France?": "Paris",
    "Capital of Japan?": "Tokyo",
    "Capital of Egypt?": "Cairo",
}


def sample_pool() -> list[Sample]:
    pool = []
    for topic, facts in (("math", MATH), ("geography", GEOGRAPHY)):
        for i, (question, answer) in enumerate(facts.items()):
            pool.append(
                Sample(
                    id=f"{topic}-{i}",
                    input=question,
                    target=answer,
                    metadata={"topic": topic},
                )
            )
    return pool


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-teacher-"))
    store = SQLiteRecordStore(workdir / "state.sqlite3")

    teacher = Teacher(
        state=KnowledgeState(store),
        exam_generator=DatasetExamGenerator(sample_pool()),
        grader=ExactMatchGrader(),
        config=TeacherConfig(confidence_smoothing=0.5, weakness_threshold=0.6),
    )

    students = [
        ScriptedStudent("student-math", "math", knowledge=MATH),
        ScriptedStudent("student-geo", "geography", knowledge=GEOGRAPHY),
    ]

    print("\n=== Round 1: initial examination ===")
    exam = teacher.create_exam(num_questions=6, seed=42)
    for student in students:
        result = teacher.evaluate(student, exam)
        print(f"{student.student_id}: {result.score:.2f}  by topic {result.topic_scores()}")

    print("\n=== Goals assigned from weaknesses ===")
    for student in students:
        for goal in teacher.assign_goals(student.student_id):
            print(f"{goal.student_id}: study {goal.topic!r} "
                  f"(priority {goal.priority:.2f}) — {goal.reason}")

    print("\n=== Students study their assigned topics ===")
    facts = {**MATH, **GEOGRAPHY}
    for student in students:
        for goal in teacher.state.current_goals(student.student_id):
            for question, answer in facts.items():
                student.learn(question, answer)
            print(f"{student.student_id} studied {goal.topic}")

    print("\n=== Round 2: re-examination ===")
    exam2 = teacher.create_exam(num_questions=6, seed=7)
    for student in students:
        result = teacher.evaluate(student, exam2)
        print(f"{student.student_id}: {result.score:.2f}")

    print("\n=== Progress reports ===")
    for student in students:
        report = teacher.progress(student.student_id)
        print(f"{student.student_id}: {report.exams_taken} exams, "
              f"mean score {report.mean_score:.2f}")
        for topic in report.topics:
            print(f"  {topic.topic}: {topic.first:.2f} -> {topic.latest:.2f} "
                  f"(delta {topic.delta:+.2f}, {topic.observations} observations)")

    print("\n=== Belief history (versioned, with reasons) ===")
    for when, confidence in teacher.state.confidence_history("student-math", "geography"):
        print(f"  {when.isoformat()}  confidence={confidence:.2f}")

    store.close()
    print(f"\nDone. State preserved at {workdir}/state.sqlite3")


if __name__ == "__main__":
    main()
