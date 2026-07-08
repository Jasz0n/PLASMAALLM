"""Tests for note snapshot used in weight-only held-out evaluation."""

from allm.models import EchoModel, ModelSpec
from allm.students import ModelStudent


def test_snapshot_and_replace_notes() -> None:
    student = ModelStudent(
        "s1",
        "fiction",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    student.study("Q1", "A1")
    backup = student.snapshot_notes()
    student.replace_notes({})
    assert student.notes == []
    student.replace_notes(backup)
    assert ("Q1", "A1") in student.notes
