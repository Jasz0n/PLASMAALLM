"""LoRA training helpers — prompt format and label masking."""

from __future__ import annotations

from allm.data.base import Sample
from allm.students.model_student import ModelStudent


def build_training_prompt(student: ModelStudent, sample: Sample) -> str:
    """Match :meth:`ModelStudent.build_prompt` without notes or confidence."""
    return "\n".join(
        [
            f"You are a student specialising in {student.specialty}.",
            "Answer the question concisely.",
            f"Question: {sample.input}",
            "Answer:",
        ]
    )


def build_training_completion(target: str) -> str:
    """Completion text (including leading space the model should emit)."""
    return f" {target.strip()}"


def mask_labels(prompt_ids: list[int], full_ids: list[int]) -> list[int]:
    """Return labels with prompt tokens masked for causal LM loss."""
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    for index in range(prompt_len):
        labels[index] = -100
    return labels
