"""Peer consultation: cross-domain failure triggers expert-routed samples.

Offline demo — software student fails a plasma question, consults
plasma-student, and receives targeted samples from the shared pool.

    PYTHONPATH=src python3 examples/24_peer_consultation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.mixed_corpus import load_mixed_corpus
from allm.students import ScriptedStudent, load_identity
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, consultation_samples, request_consultation

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    corpus = load_mixed_corpus(ROOT)
    plasma_id = load_identity(ROOT / "configs/students/plasma_student.yaml")
    software_id = load_identity(ROOT / "configs/students/software_student.yaml")

    pool = SamplePool()
    pool.ingest(corpus.merged_train)

    workdir = Path(tempfile.mkdtemp(prefix="allm-peer-consult-"))
    store = SQLiteRecordStore(workdir / "state.sqlite3")
    state = KnowledgeState(store)

    plasma_q = next(s for s in corpus.plasma.holdout if s.target)
    software_q = next(s for s in corpus.software.holdout if s.target)
    exam_samples = [
        Sample(id="cross-1", input=plasma_q.input, target=plasma_q.target, metadata=plasma_q.metadata),
        Sample(id="cross-2", input=software_q.input, target=software_q.target, metadata=software_q.metadata),
    ]
    teacher = Teacher(state, DatasetExamGenerator(exam_samples), ExactMatchGrader("contains"))

    teacher.evaluate(
        ScriptedStudent(
            plasma_id.student_id,
            "kids-plasma",
            knowledge={plasma_q.input: plasma_q.target},
        ),
        teacher.create_exam(num_questions=1, seed=1),
    )
    cross_exam = teacher.create_exam(num_questions=2, seed=2)
    result = teacher.evaluate(
        ScriptedStudent(software_id.student_id, "fastify-api", knowledge={}),
        cross_exam,
    )

    print("\n=== Cross-domain exam (software student) ===")
    print(f"  score: {result.score:.2f}  failures: {len(result.failures())}")

    for failure in result.failures():
        topic = failure.question.topic
        req = request_consultation(state, software_id.student_id, topic)
        print(f"  failed {topic!r} -> consult {req.expert_id or 'none'} ({req.reason})")

    peer_samples, requests = consultation_samples(
        state,
        pool,
        software_id.student_id,
        software_id,
        result,
        mission_seed=42,
        samples_per_topic=3,
    )
    print(f"\n=== Peer teaching ===")
    print(f"  consultations: {len(requests)}")
    print(f"  samples received: {len(peer_samples)}")
    for sample in peer_samples[:5]:
        print(f"    - [{sample.metadata.get('topic')}] {sample.input[:60]}")

    store.close()
    print(f"\nDone. State at {workdir}")


if __name__ == "__main__":
    main()
