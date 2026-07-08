"""Tests for allm.trainer (in-context backend)."""

from allm.data.base import Sample
from allm.models import EchoModel, ModelSpec
from allm.students import ModelStudent
from allm.trainer import InContextTrainer, Trainer, trainers


def student() -> ModelStudent:
    spec = ModelSpec(name="m", provider="echo", model_id="none")
    return ModelStudent("s1", "math", EchoModel(spec))


def test_satisfies_protocol_and_registered() -> None:
    assert isinstance(InContextTrainer(), Trainer)
    assert trainers.get("in_context") is InContextTrainer
    assert trainers.get("lora") is not None


def test_train_studies_labelled_samples() -> None:
    learner = student()
    report = InContextTrainer().train(
        learner,
        [
            Sample(id="a", input="2+2?", target="4"),
            Sample(id="b", input="open question", target=None),
        ],
    )
    assert report.samples_used == 1
    assert report.samples_skipped == 1
    assert report.method == "in_context"
    assert ("2+2?", "4") in learner.notes


def test_train_empty_is_harmless() -> None:
    report = InContextTrainer().train(student(), [])
    assert report.samples_used == 0
