"""Ecosystem targeting — per-student Researcher recommendations (offline).

    PYTHONPATH=src python3 examples/32_ecosystem_targeting.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.core.logging import setup_logging
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.mixed_corpus import load_mixed_corpus
from allm.knowledge import Concept, KnowledgeGraph
from allm.researcher import ResearcherLayer
from allm.students import load_identity
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from dual_consult_run import _make_student

STUDENTS = ROOT / "configs/students"


def main() -> None:
    setup_logging("INFO")
    corpus = load_mixed_corpus(ROOT)
    plasma_id = load_identity(STUDENTS / "plasma_student.yaml")
    software_id = load_identity(STUDENTS / "software_student.yaml")

    workdir = Path(tempfile.mkdtemp(prefix="allm-eco-target-"))
    store = SQLiteRecordStore(workdir / "state.sqlite3")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="kids-plasma", description="plasma"))
    graph.add(Concept(name="fastify-api", description="software"))

    state = KnowledgeState(store)
    plasma_student = _make_student(plasma_id.student_id, "kids-plasma", list(corpus.plasma.train))
    software_student = _make_student(software_id.student_id, "fastify-api", list(corpus.software.train))
    teacher = Teacher(
        state,
        DatasetExamGenerator(list(corpus.plasma.holdout[:2]) + list(corpus.software.holdout[:2])),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )
    teacher.evaluate(plasma_student, teacher.create_exam(num_questions=2, seed=1))
    teacher.evaluate(software_student, teacher.create_exam(num_questions=2, seed=2))

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        software_samples=ROOT / "transcripts/Software/samples_dev.jsonl",
        workshop_max_files=1,
        catalog_topics=("kids-plasma", "fastify-api", "prisma-orm", "typescript-react"),
        graph=graph,
        state=state,
        identities={
            plasma_id.student_id: plasma_id,
            software_id.student_id: software_id,
        },
        student_ids=(plasma_id.student_id, software_id.student_id),
    )
    report = researcher.run_cycle()
    ecosystem = researcher.ecosystem_metrics(graph, state)

    print("\n=== Ecosystem targeting ===")
    print(f"  recommendations total: {len(report.recommendations)}")
    plasma_recs = researcher.active_recommendations(student_id=plasma_id.student_id)
    software_recs = researcher.active_recommendations(student_id=software_id.student_id)
    print(f"  plasma-student recs: {len(plasma_recs)} topics={sorted({r.topic for r in plasma_recs})}")
    print(f"  software-student recs: {len(software_recs)} topics={sorted({r.topic for r in software_recs})}")

    if ecosystem.student_topic_matrix is not None:
        matrix = ecosystem.student_topic_matrix
        for sid in (plasma_id.student_id, software_id.student_id):
            row = matrix.rows.get(sid, {})
            print(f"  matrix {sid}: {dict(list(row.items())[:4])}")

    os.environ["ALLM_MEDIATED_CONSULT"] = "0"
    os.environ["ALLM_RESEARCHER_TARGETING"] = "1"
    print("\n  (set ALLM_RESEARCHER_TARGETING=1 in loop for per-student catalog boost)")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
