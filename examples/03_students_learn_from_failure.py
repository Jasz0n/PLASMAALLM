"""Phase 3 demo: the first closed learning cycle.

A model-backed student takes an exam, fails, its failures are stored
(versioned) and converted back into training samples, a trainer teaches
it, and the re-examination shows measured improvement.

Runs entirely offline (echo model, temp storage):

    python examples/03_students_learn_from_failure.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.models import EchoModel, ModelSpec
from allm.students import FailureLog, ModelStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import InContextTrainer

FACTS = {
    "Capital of France?": ("Paris", "geography"),
    "Capital of Japan?": ("Tokyo", "geography"),
    "2+2?": ("4", "math"),
    "3*3?": ("9", "math"),
}


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-phase3-"))
    store = SQLiteRecordStore(workdir / "state.sqlite3")

    samples = [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, (a, topic)) in enumerate(FACTS.items())
    ]
    teacher = Teacher(
        state=KnowledgeState(store),
        exam_generator=DatasetExamGenerator(samples),
        grader=ExactMatchGrader(),
        config=TeacherConfig(confidence_smoothing=0.5),
    )
    failure_log = FailureLog(store)

    # A model-backed student over a (deliberately clueless) echo model.
    model = EchoModel(ModelSpec(name="blank", provider="echo", model_id="none"))
    student = ModelStudent("student-1", "general knowledge", model)

    print("\n=== 1. Measure: initial exam ===")
    exam = teacher.create_exam(num_questions=4, seed=1)
    result = teacher.evaluate(student, exam)
    print(f"score {result.score:.2f}; topic scores {result.topic_scores()}")

    print("\n=== 2. Store failures (Plan.md: failure is training data) ===")
    for failed in result.failures():
        record = failure_log.record(student.student_id, failed)
        print(f"failed {record.question_id}: gave {record.given!r}, "
              f"expected {record.expected!r} (confidence {record.confidence:.2f})")

    print("\n=== 3. Learn: train on own failures ===")
    report = InContextTrainer().train(
        student, failure_log.training_samples(student.student_id)
    )
    print(f"studied {report.samples_used} failure-derived sample(s)")

    print("\n=== 4. Re-measure ===")
    exam2 = teacher.create_exam(num_questions=4, seed=2)
    result2 = teacher.evaluate(student, exam2)
    print(f"score {result2.score:.2f}")

    print("\n=== 5. Progress (from versioned history) ===")
    progress = teacher.progress(student.student_id)
    for topic in progress.topics:
        print(f"  {topic.topic}: {topic.first:.2f} -> {topic.latest:.2f} "
              f"(delta {topic.delta:+.2f})")

    store.close()
    print(f"\nDone. State preserved at {workdir}/state.sqlite3")


if __name__ == "__main__":
    main()
