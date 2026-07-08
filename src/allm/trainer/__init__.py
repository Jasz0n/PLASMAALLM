"""Trainers: how students learn from labelled samples.

Phase 3 ships in-context learning (note-taking). Weight-level
fine-tuning (LoRA via peft) is a planned second backend behind the same
:class:`~allm.trainer.base.Trainer` protocol.
"""

from allm.trainer.base import Trainer, TrainingReport, trainers
from allm.trainer.adapters import AdapterRecord, AdapterStore
from allm.trainer.forgetting import ForgettingReport, ForgettingWatchdog
from allm.trainer.in_context import InContextTrainer
from allm.trainer.lora import LoRAConfig, LoRATrainer

__all__ = [
    "AdapterRecord",
    "AdapterStore",
    "ForgettingReport",
    "ForgettingWatchdog",
    "LoRAConfig",
    "LoRATrainer",
    "Trainer",
    "TrainingReport",
    "trainers",
    "InContextTrainer",
]
