"""Hugging Face ``datasets`` loader.

Like the HF model loader, the heavy import happens inside ``load`` so
this module (and the ``huggingface`` source name) is always available;
using it without the ML extras raises a clear error.

``spec.options`` may contain:
- ``input_field`` / ``target_field``: column names (default
  ``"input"`` / ``"target"``)
- any other keys are passed through to ``datasets.load_dataset``.
"""

from __future__ import annotations

from typing import Iterator

from allm.data.base import DatasetSpec, Sample, dataset_loaders


@dataset_loaders.register("huggingface")
class HFDatasetLoader:
    """Streams samples from a Hugging Face hub dataset."""

    def load(self, spec: DatasetSpec) -> Iterator[Sample]:
        try:
            import datasets
        except ImportError as exc:
            raise ImportError(
                "the 'huggingface' dataset source needs the ML extras: pip install -e '.[ml]'"
            ) from exc

        options = dict(spec.options)
        input_field = options.pop("input_field", "input")
        target_field = options.pop("target_field", "target")
        ds = datasets.load_dataset(spec.location, split=spec.split, **options)
        for index, row in enumerate(ds):
            row = dict(row)
            if input_field not in row:
                raise ValueError(
                    f"dataset {spec.name!r} has no column {input_field!r}; "
                    f"set options.input_field (columns: {sorted(row)})"
                )
            yield Sample(
                id=f"{spec.name}-{index}",
                input=str(row.pop(input_field)),
                target=None if target_field not in row else str(row.pop(target_field)),
                metadata=row,
            )
