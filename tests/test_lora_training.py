"""Tests for LoRA training prompt alignment (offline)."""

from allm.data.base import Sample
from allm.models import EchoModel, ModelSpec
from allm.students import ModelStudent
from allm.trainer.lora_training import (
    build_training_completion,
    build_training_prompt,
    mask_labels,
)


def test_training_prompt_matches_exam_shape() -> None:
    student = ModelStudent(
        "s1",
        "fiction",
        EchoModel(ModelSpec(name="m", provider="echo", model_id="none")),
    )
    sample = Sample(id="a", input="What is Project Zeta?", target="Nebula")
    prompt = build_training_prompt(student, sample)
    assert "specialising in fiction" in prompt
    assert "Question: What is Project Zeta?" in prompt
    assert prompt.endswith("Answer:")
    assert "CONFIDENCE" not in prompt


def test_mask_labels_hides_prompt_tokens() -> None:
    prompt_ids = [1, 2, 3]
    full_ids = [1, 2, 3, 4, 5]
    labels = mask_labels(prompt_ids, full_ids)
    assert labels[:3] == [-100, -100, -100]
    assert labels[3:] == [4, 5]


def test_build_training_completion() -> None:
    assert build_training_completion("Nebula") == " Nebula"
