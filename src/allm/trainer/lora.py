"""LoRA/PEFT trainer: weight-level learning for Hugging Face students.

Requires ``pip install -e '.[ml]'`` and a :class:`~allm.models.huggingface.HFModel`
student. Ollama-backed students should use :class:`InContextTrainer` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.data.base import Sample
from allm.models.huggingface import HFModel
from allm.students.model_student import ModelStudent
from allm.trainer.adapters import AdapterStore
from allm.trainer.base import TrainingReport, trainers
from allm.trainer.lora_training import (
    build_training_completion,
    build_training_prompt,
    mask_labels,
)

logger = get_logger("trainer.lora")


class LoRAConfig(BaseModel):
    """Tunables for a LoRA fine-tune."""

    model_config = ConfigDict(frozen=True)

    r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    epochs: int = Field(default=5, ge=1)
    repetitions: int = Field(default=3, ge=1, description="Passes over each sample per epoch")
    learning_rate: float = Field(default=3e-4, gt=0)
    max_seq_length: int = Field(default=256, ge=32)
    target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")


def _run_lora_training(
    student: ModelStudent,
    samples: list[Sample],
    config: LoRAConfig,
    output_dir: Path,
) -> None:
    """Execute LoRA fine-tune with exam-aligned prompts and answer-only loss."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.optim import AdamW
    except ImportError as exc:
        raise ImportError(
            "LoRATrainer needs ML extras with peft: pip install -e '.[ml]'"
        ) from exc

    model = student._model
    assert isinstance(model, HFModel)
    tokenizer = model._tokenizer
    base = model._model
    if not hasattr(base, "peft_config"):
        peft_cfg = LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            task_type="CAUSAL_LM",
        )
        trainable = get_peft_model(base, peft_cfg)
    else:
        trainable = base

    optimizer = AdamW(trainable.parameters(), lr=config.learning_rate)
    trainable.train()
    for epoch in range(config.epochs):
        for sample in samples:
            prompt = build_training_prompt(student, sample)
            completion = build_training_completion(sample.target or "")
            for _ in range(config.repetitions):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                full_text = prompt + completion
                encodings = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=config.max_seq_length,
                    return_tensors="pt",
                )
                input_ids = encodings["input_ids"].to(trainable.device)
                labels = torch.tensor(
                    [mask_labels(prompt_ids, input_ids[0].tolist())],
                    device=trainable.device,
                )
                outputs = trainable(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                logger.info(
                    "lora epoch %d loss=%.4f sample=%r",
                    epoch + 1,
                    float(loss.item()),
                    sample.input[:40],
                )

    trainable.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable.save_pretrained(output_dir)
    model._model = trainable


@trainers.register("lora")
class LoRATrainer:
    """Fine-tunes LoRA adapters on failure-derived samples."""

    def __init__(
        self,
        adapter_store: AdapterStore,
        config: LoRAConfig | None = None,
    ) -> None:
        self._adapters = adapter_store
        self._config = config or LoRAConfig()

    def train(self, student: ModelStudent, samples: Iterable[Sample]) -> TrainingReport:
        labelled = [sample for sample in samples if sample.target is not None]
        skipped = sum(1 for sample in samples if sample.target is None)
        model = student._model
        if not isinstance(model, HFModel):
            raise TypeError(
                "LoRATrainer requires a Hugging Face student; "
                "use InContextTrainer for Ollama or echo models"
            )
        if not labelled:
            logger.info("%s: no labelled samples for lora", student.student_id)
            return TrainingReport(
                student_id=student.student_id,
                method="lora",
                samples_used=0,
                samples_skipped=skipped,
            )

        workdir = self._adapters.scratch_dir(student.student_id)
        import shutil

        if workdir.exists() and any(workdir.iterdir()):
            shutil.rmtree(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
        _run_lora_training(student, labelled, self._config, workdir)
        adapter_id = self._adapters.save(
            student.student_id,
            workdir,
            base_model_id=model.spec.model_id,
            samples_trained=len(labelled),
            reason=f"lora fine-tune on {len(labelled)} sample(s)",
        )
        logger.info(
            "%s lora trained on %d sample(s), adapter %s",
            student.student_id,
            len(labelled),
            adapter_id,
        )
        student.set_adapter(adapter_id)
        return TrainingReport(
            student_id=student.student_id,
            method="lora",
            samples_used=len(labelled),
            samples_skipped=skipped,
            adapter_id=adapter_id,
        )
