"""Teacher-mediated consultation demo (offline).

    PYTHONPATH=src python3 examples/26_mediated_consultation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.mixed_corpus import load_mixed_corpus
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, mediated_consultation

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    corpus = load_mixed_corpus(ROOT)
    plasma_q = next(s for s in corpus.plasma.holdout if s.target)

    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    teacher = Teacher(
        state,
        DatasetExamGenerator([plasma_q]),
        ExactMatchGrader("contains"),
    )
    teacher.evaluate(
        ScriptedStudent("plasma-student", "kids-plasma", knowledge={plasma_q.input: plasma_q.target}),
        teacher.create_exam(num_questions=1, seed=1),
    )

    expert = ScriptedStudent("plasma-student", "kids-plasma", knowledge={plasma_q.input: plasma_q.target})
    result = mediated_consultation(
        state,
        ExactMatchGrader("contains"),
        "software-student",
        expert,
        topic=str(plasma_q.metadata.get("topic", "kids-plasma")),
        prompt=plasma_q.input,
        expected=plasma_q.target or "",
    )

    print("\n=== Teacher-mediated consultation ===")
    print(f"  asker: software-student")
    print(f"  expert: {result.expert_id}")
    print(f"  topic: {result.topic}")
    print(f"  approved: {result.approved}")
    print(f"  reason: {result.reason}")
    if result.study_sample:
        print(f"  approved sample: {result.study_sample.input[:60]}...")

    store.close()


if __name__ == "__main__":
    main()
