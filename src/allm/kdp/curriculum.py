"""Load Kids curriculum samples with optional kind filter and hold-out split."""

from __future__ import annotations

import os
from pathlib import Path

from allm.data.base import Sample
from allm.kdp.corpus import load_samples_jsonl
from allm.kdp.holdout import split_samples_holdout
from allm.kdp.mk_samples import filter_samples_by_kind, parse_sample_kinds

DEFAULT_EXAM_PATH = Path("transcripts/Kids/samples_exam.jsonl")
DEFAULT_DEFINITIONS_PATH = Path("transcripts/Kids/samples_definitions.jsonl")


def resolve_samples_path(root: Path) -> Path:
    """Pick jsonl from env or default exam pool."""
    override = os.environ.get("ALLM_SAMPLES_FILE", "").strip()
    if override:
        return Path(override)
    if os.environ.get("ALLM_SAMPLES", "").lower() == "definitions":
        path = root / DEFAULT_DEFINITIONS_PATH
        if path.is_file():
            return path
    return root / DEFAULT_EXAM_PATH


def load_curriculum_splits(
    root: Path,
    *,
    holdout_after: int | None = None,
) -> tuple[list[Sample], list[Sample]]:
    """Load labelled samples, optional kind filter, then workshop hold-out."""
    path = resolve_samples_path(root)
    if not path.is_file():
        raise FileNotFoundError(path)
    samples = [s for s in load_samples_jsonl(path) if s.target]
    kind_env = os.environ.get("ALLM_SAMPLE_KIND", "").strip()
    if kind_env:
        kinds = parse_sample_kinds(kind_env)
        samples = filter_samples_by_kind(samples, kinds)
    if holdout_after is None:
        holdout_after = int(os.environ.get("ALLM_HOLDOUT_AFTER", "13"))
    train, holdout = split_samples_holdout(samples, holdout_after=holdout_after)
    return train, holdout
