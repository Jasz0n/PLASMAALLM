"""Tests for mixed corpus loading and specialist sample filtering."""

from pathlib import Path

import pytest

from allm.kdp.mixed_corpus import load_mixed_corpus, samples_for_identity, split_samples
from allm.students import StudentIdentity, load_identity
from allm.teacher import request_consultation
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher

ROOT = Path(__file__).resolve().parents[1]


def test_split_samples_holdout() -> None:
    samples = [
        Sample(id=f"s{i}", input=f"q{i}", target=f"a{i}", metadata={"topic": "t"})
        for i in range(8)
    ]
    train, holdout = split_samples(samples, holdout_fraction=0.25, seed=1)
    assert len(holdout) == 2
    assert len(train) == 6


def test_samples_for_identity_filters_domains() -> None:
    plasma = StudentIdentity(
        student_id="plasma-student",
        primary_domains=("plasma",),
        exploration_rate=0.0,
    )
    samples = [
        Sample(id="p", input="q", target="a", metadata={"topic": "kids-plasma"}),
        Sample(id="s", input="q2", target="a2", metadata={"topic": "fastify-api"}),
    ]
    kept = samples_for_identity(samples, plasma)
    assert len(kept) == 1
    assert kept[0].id == "p"


def test_load_mixed_corpus() -> None:
    corpus = load_mixed_corpus(ROOT)
    assert len(corpus.plasma.train) > 0
    assert len(corpus.software.train) > 0
    assert len(corpus.merged_train) == len(corpus.plasma.train) + len(corpus.software.train)


def test_request_consultation(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "s.sqlite3")
    state = KnowledgeState(store)
    samples = [
        Sample(id="p1", input="Plasma?", target="fields", metadata={"topic": "kids-plasma"}),
    ]
    teacher = Teacher(state, DatasetExamGenerator(samples), ExactMatchGrader("contains"))
    teacher.evaluate(
        ScriptedStudent("plasma-student", "kids-plasma", knowledge={"Plasma?": "fields"}),
        teacher.create_exam(num_questions=1, seed=1),
    )
    req = request_consultation(state, "software-student", "kids-plasma")
    assert req.expert_id == "plasma-student"
    store.close()


def test_consultation_samples_from_pool() -> None:
    from allm.collector import SamplePool
    from allm.exam import DatasetExamGenerator, ExactMatchGrader
    from allm.teacher import Teacher, consultation_samples

    plasma = load_identity(ROOT / "configs/students/plasma_student.yaml")
    software = load_identity(ROOT / "configs/students/software_student.yaml")
    pool = SamplePool()
    plasma_sample = Sample(
        id="p1",
        input="What is plasma?",
        target="fields",
        metadata={"topic": "kids-plasma"},
    )
    pool.ingest([plasma_sample])

    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    teacher = Teacher(state, DatasetExamGenerator([plasma_sample]), ExactMatchGrader("contains"))
    teacher.evaluate(
        ScriptedStudent("plasma-student", "kids-plasma", knowledge={"What is plasma?": "fields"}),
        teacher.create_exam(num_questions=1, seed=1),
    )
    cross = teacher.create_exam(num_questions=1, seed=2)
    result = teacher.evaluate(ScriptedStudent("software-student", "fastify", knowledge={}), cross)

    samples, requests = consultation_samples(
        state, pool, "software-student", software, result, mission_seed=1
    )
    assert requests[0].expert_id == "plasma-student"
    assert len(samples) >= 1
    store.close()
