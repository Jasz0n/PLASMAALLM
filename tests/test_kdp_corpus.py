"""Tests for cleaned corpus loading and KU -> Sample conversion."""

import json
from pathlib import Path

from allm.data.base import Sample
from allm.kdp.corpus import export_samples_jsonl, load_samples_jsonl, unit_to_sample, units_to_samples
from allm.kdp.types import KnowledgeUnit

CONCEPT = KnowledgeUnit(
    id="ku_plasma_001",
    type="concept",
    content="Plasma is a rotating sphere of magnetic fields.",
    normalized_concept="Plasma",
    confidence=0.9,
    sources=("workshop3.txt",),
    context="kids-plasma",
    raw_span_refs=(),
)


def test_unit_to_sample_concept() -> None:
    sample = unit_to_sample(CONCEPT)
    assert sample is not None
    assert sample.input == "What is Plasma?"
    assert "rotating sphere" in sample.target or ""


def test_units_to_samples_skips_unlabelled_questions() -> None:
    question = CONCEPT.model_copy(
        update={"id": "q1", "type": "question", "content": "What is gravity?"}
    )
    samples = units_to_samples([CONCEPT, question], labelled_only=True)
    assert len(samples) == 1
    assert samples[0].id == CONCEPT.id


def test_load_samples_jsonl_roundtrip(tmp_path: Path) -> None:
    sample = Sample(id="s1", input="What is Plasma?", target="A sphere of fields.", metadata={"topic": "kids-plasma"})
    path = tmp_path / "samples.jsonl"
    export_samples_jsonl([sample], path)
    loaded = load_samples_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].input == sample.input
    assert loaded[0].metadata["topic"] == "kids-plasma"


def test_load_kids_samples_file() -> None:
    path = Path(__file__).resolve().parents[1] / "transcripts" / "Kids" / "samples.jsonl"
    if not path.is_file():
        return
    samples = load_samples_jsonl(path)
    assert len(samples) >= 100
    assert all(s.target for s in samples)
