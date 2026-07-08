"""Workshop-level train / hold-out splits for Kids curriculum samples."""

from __future__ import annotations

import re

from allm.data.base import Sample

_WORKSHOP_NUM = re.compile(r"workshop\s*(\d+)", re.IGNORECASE)


def sample_source(sample: Sample) -> str:
    """Source transcript filename from sample metadata."""
    return str((sample.metadata or {}).get("source", ""))


def workshop_number(source: str) -> int:
    """Parse workshop index from filenames like ``knowledgeSeekerWorkshop13.txt``."""
    match = _WORKSHOP_NUM.search(source)
    return int(match.group(1)) if match else 0


def split_samples_holdout(
    samples: list[Sample],
    *,
    holdout_after: int = 13,
) -> tuple[list[Sample], list[Sample]]:
    """Split by workshop number: train on ``< holdout_after``, test on ``>=``.

    Default ``holdout_after=13`` → train workshops 1–12, held-out 13–22.
    """
    train: list[Sample] = []
    holdout: list[Sample] = []
    for sample in samples:
        number = workshop_number(sample_source(sample))
        if number <= 0:
            train.append(sample)
        elif number >= holdout_after:
            holdout.append(sample)
        else:
            train.append(sample)
    return train, holdout
