"""Tests for the forgetting watchdog."""

from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.models import EchoModel, ModelSpec
from allm.students import ModelStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import ForgettingWatchdog


def test_forgetting_detects_regression() -> None:
    store = SQLiteRecordStore(":memory:")
    samples = [
        Sample(id="s0", input="2+2?", target="4", metadata={"topic": "math"}),
        Sample(id="s1", input="Capital of France?", target="Paris", metadata={"topic": "geo"}),
    ]
    teacher = Teacher(
        KnowledgeState(store),
        DatasetExamGenerator(samples),
        ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=1.0),
    )
    student = ModelStudent(
        "s1",
        "general",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    student.study("2+2?", "4")
    exam = teacher.create_exam(topics=["math"], num_questions=1, seed=1)
    teacher.evaluate(student, exam)

    watchdog = ForgettingWatchdog(teacher, mastery_threshold=0.5, regression_threshold=0.1)
    mastered = watchdog.mastered_topics(teacher.state, "s1")
    assert "math" in mastered

    report = watchdog.check(student, mastered, seed=99)
    assert report.student_id == "s1"
    assert "math" in report.probed_topics
    store.close()


def test_lora_trainer_rejects_non_hf_student() -> None:
    import pytest
    from pathlib import Path

    from allm.trainer import AdapterStore, LoRATrainer
    from allm.data.base import Sample

    store = SQLiteRecordStore(":memory:")
    adapters = AdapterStore(store, Path("/tmp/allm-test-adapters"))
    trainer = LoRATrainer(adapters)
    student = ModelStudent(
        "s1",
        "math",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    with pytest.raises(TypeError, match="Hugging Face"):
        trainer.train(student, [Sample(id="a", input="2+2?", target="4")])
    store.close()
