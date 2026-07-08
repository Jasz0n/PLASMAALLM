"""Load cleaned transcript corpora and bridge KDP units to training samples."""

from __future__ import annotations

import json
from pathlib import Path

from allm.data.base import Sample
from allm.kdp.ingestion import DocumentStore
from allm.kdp.types import KnowledgeUnit

DEFAULT_TOPIC = "kids-plasma"


def ingest_cleaned_corpus(
    store: DocumentStore,
    directory: Path | str,
    *,
    pattern: str = "*.txt",
    context: str = DEFAULT_TOPIC,
) -> list:
    """Ingest cleaned transcript exports (prose paragraphs, no timestamps)."""
    return store.ingest_directory(directory, pattern=pattern, context=context)


def _answer_from_unit(unit: KnowledgeUnit) -> str:
    """First perspective line or content head as the reference answer."""
    for line in unit.content.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("--- perspective ---"):
            continue
        if cleaned:
            return cleaned[:400]
    return unit.content[:400].strip()


def unit_to_sample(unit: KnowledgeUnit, *, topic: str = DEFAULT_TOPIC) -> Sample | None:
    """Turn one knowledge unit into a labelled training/exam sample."""
    meta = {
        "topic": topic,
        "ku_type": unit.type,
        "sources": list(unit.sources),
        "confidence": unit.confidence,
    }
    answer = _answer_from_unit(unit)
    if unit.type == "question":
        prompt = unit.content.strip()
        if not prompt.endswith("?"):
            prompt = f"{prompt}?"
        return Sample(id=unit.id, input=prompt, target=None, metadata=meta)
    if unit.type == "misconception":
        prompt = f"What is wrong about how people think about {unit.normalized_concept}?"
    elif unit.type == "procedure":
        prompt = f"How do you {unit.normalized_concept}?"
    else:
        prompt = f"What is {unit.normalized_concept}?"
    return Sample(id=unit.id, input=prompt, target=answer, metadata=meta)


def units_to_samples(
    units: list[KnowledgeUnit],
    *,
    topic: str = DEFAULT_TOPIC,
    labelled_only: bool = True,
) -> list[Sample]:
    """Convert distilled knowledge units into :class:`Sample` rows."""
    samples: list[Sample] = []
    for unit in units:
        sample = unit_to_sample(unit, topic=topic)
        if sample is None:
            continue
        if labelled_only and sample.target is None:
            continue
        samples.append(sample)
    return samples


def export_samples_jsonl(samples: list[Sample], path: Path | str) -> int:
    """Write samples for ``SamplePool`` / dataset loaders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            row = {
                "id": sample.id,
                "input": sample.input,
                "target": sample.target,
                **sample.metadata,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(samples)


def load_samples_jsonl(path: Path | str) -> list[Sample]:
    """Load labelled samples exported by :func:`export_samples_jsonl`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "input" not in raw:
                raise ValueError(f"{path}:{lineno}: missing 'input'")
            meta = {k: v for k, v in raw.items() if k not in ("id", "input", "target")}
            samples.append(
                Sample(
                    id=str(raw.get("id", f"{path.stem}-{lineno}")),
                    input=raw["input"],
                    target=raw.get("target"),
                    metadata=meta,
                )
            )
    return samples
