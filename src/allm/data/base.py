"""Dataset interfaces.

Design decisions
----------------
- Every loader yields the same normalised :class:`Sample` shape
  (id / input / target / metadata) regardless of source. Exams, the
  trainer and the data collector (later phases) then only deal with
  one shape, and new sources are pure adapters.
- Loaders return iterators, not lists: corpora may not fit in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from allm.core.registry import Registry


class Sample(BaseModel):
    """One normalised task/example."""

    model_config = ConfigDict(frozen=True)

    id: str
    input: str
    target: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetSpec(BaseModel):
    """Declarative description of a dataset, loadable from YAML.

    ``source`` selects the loader from :data:`dataset_loaders`;
    ``location`` is interpreted by that loader (file path, hub id, ...).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    source: str = "jsonl"
    location: str
    split: str = "train"
    options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DatasetSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


@runtime_checkable
class DatasetLoader(Protocol):
    """Turns a :class:`DatasetSpec` into a stream of samples."""

    def load(self, spec: DatasetSpec) -> Iterator[Sample]: ...


dataset_loaders: Registry[type] = Registry("dataset_loader")


def load_dataset(spec: DatasetSpec) -> Iterator[Sample]:
    """Convenience: pick the loader for ``spec.source`` and load."""
    loader_cls = dataset_loaders.get(spec.source)
    return loader_cls().load(spec)
