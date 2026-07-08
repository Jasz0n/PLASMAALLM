"""Tests for adapter provenance on exam results."""

from allm.exam.base import Answer, ExamResult, Question, QuestionResult
from allm.models import EchoModel, ModelSpec
from allm.students import ModelStudent


def test_exam_result_carries_adapter_id() -> None:
    student = ModelStudent(
        "s1",
        "math",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    student.set_adapter("s1-lora-0001")
    q = Question(id="q1", prompt="2+2?", expected="4", topic="math")
    result = ExamResult(
        exam_id="e1",
        student_id="s1",
        adapter_id=student.active_adapter_id,
        results=(
            QuestionResult(
                question=q,
                answer=Answer(question_id="q1", text="4", confidence=0.9),
                score=1.0,
                correct=True,
            ),
        ),
    )
    assert result.adapter_id == "s1-lora-0001"
