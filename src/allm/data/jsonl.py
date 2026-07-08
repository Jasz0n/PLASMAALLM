"""JSONL dataset loader.

The default local format: one JSON object per line with at least an
``input`` field; ``id``, ``target`` and any extra keys are optional
(extras land in ``Sample.metadata``). Lines are streamed, never fully
materialised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from allm.data.base import DatasetSpec, Sample, dataset_loaders


@dataset_loaders.register("jsonl")
class JSONLDatasetLoader:
    """Streams samples from a local ``.jsonl`` file."""

    def load(self, spec: DatasetSpec) -> Iterator[Sample]:
        path = Path(spec.location)
        if not path.exists():
            raise FileNotFoundError(f"dataset {spec.name!r}: no such file {path}")
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if "input" not in raw:
                    raise ValueError(f"{path}:{lineno}: sample is missing the 'input' field")
                yield Sample(
                    id=str(raw.pop("id", f"{spec.name}-{lineno}")),
                    input=raw.pop("input"),
                    target=raw.pop("target", None),
                    metadata=raw,
                )
