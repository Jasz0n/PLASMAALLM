"""Dataset interfaces and loaders."""

from allm.data.base import DatasetSpec, Sample, dataset_loaders, load_dataset
from allm.data.jsonl import JSONLDatasetLoader

# Registers the "huggingface" dataset loader; importable without `datasets`.
from allm.data import huggingface as _huggingface  # noqa: F401

__all__ = ["DatasetSpec", "Sample", "dataset_loaders", "load_dataset", "JSONLDatasetLoader"]
