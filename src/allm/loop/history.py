"""Append-only learning iteration history (research dataset)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from allm.loop.strategy import LearningStrategyName


class LearningIterationRecord(BaseModel):
    """One loop iteration — suitable for JSONL export and meta-analysis."""

    model_config = ConfigDict(frozen=True)

    iteration: int
    strategy: LearningStrategyName
    sample_kinds: tuple[str, ...]
    sample_ids: tuple[str, ...]
    student_id: str
    score_before: float
    score_after: float
    goals: tuple[str, ...]
    samples_studied: int
    questions_per_exam: int
    use_exam_paraphrase: bool
    kel_lg: float | None = None
    kel_ghs: float | None = None
    kel_findings: tuple[str, ...] = ()
    failure_prompts: tuple[str, ...] = ()
    holdout_count: int | None = None
    holdout_answers_in_train: int | None = None
    holdout_novel_lexical: int | None = None
    strategy_previous: LearningStrategyName | None = None
    strategy_changed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LearningRunManifest(BaseModel):
    """One-time run metadata written before the loop starts."""

    model_config = ConfigDict(frozen=True)

    student_model: str
    train_count: int
    holdout_count: int
    holdout_exact_prompt_matches: int
    holdout_high_overlap: int
    holdout_low_overlap: int
    holdout_novel_lexical: int
    holdout_answers_in_train: int
    holdout_by_workshop: dict[int, int] = Field(default_factory=dict)
    kel_mastery_threshold: float
    kel_strategy_advance_threshold: float


class IterationHistoryWriter:
    """Append iteration records to a JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: LearningIterationRecord) -> None:
        """Write one record as a single JSONL line."""
        line = record.model_dump_json()
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    @property
    def path(self) -> Path:
        return self._path

    def load_all(self) -> list[LearningIterationRecord]:
        """Read every record from disk."""
        if not self._path.is_file():
            return []
        records: list[LearningIterationRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(LearningIterationRecord.model_validate(json.loads(line)))
        return records

    def write_manifest(self, manifest: LearningRunManifest) -> Path:
        """Write run-level curriculum and config snapshot next to the JSONL."""
        manifest_path = self._path.with_name("learning_run.json")
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest_path
