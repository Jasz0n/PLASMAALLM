"""Tests for Teacher-mediated consultation."""

from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.data.base import Sample
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, mediated_consultation


def test_mediated_consultation_approves_expert() -> None:
    sample = Sample(
        id="p1",
        input="What is plasma?",
        target="fields of energy",
        metadata={"topic": "kids-plasma"},
    )
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    teacher = Teacher(state, DatasetExamGenerator([sample]), ExactMatchGrader("contains"))
    teacher.evaluate(
        ScriptedStudent("plasma-student", "kids-plasma", knowledge={"What is plasma?": "fields of energy"}),
        teacher.create_exam(num_questions=1, seed=1),
    )
    expert = ScriptedStudent("plasma-student", "kids-plasma", knowledge={"What is plasma?": "fields of energy"})
    result = mediated_consultation(
        state,
        ExactMatchGrader("contains"),
        "software-student",
        expert,
        topic="kids-plasma",
        prompt="What is plasma?",
        expected="fields of energy",
    )
    assert result.approved
    assert result.study_sample is not None


def test_mediated_consultation_rejects_wrong_expert() -> None:
    sample = Sample(
        id="p1",
        input="What is plasma?",
        target="fields",
        metadata={"topic": "kids-plasma"},
    )
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    expert = ScriptedStudent("software-student", "fastify", knowledge={})
    result = mediated_consultation(
        state,
        ExactMatchGrader("contains"),
        "software-student",
        expert,
        topic="kids-plasma",
        prompt="What is plasma?",
        expected="fields",
    )
    assert not result.approved
