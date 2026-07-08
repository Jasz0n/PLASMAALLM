"""Tests for allm.data (jsonl backend + spec plumbing; no downloads)."""

import json
from pathlib import Path

import pytest

from allm.data import DatasetSpec, dataset_loaders, load_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_sources_registered() -> None:
    assert "jsonl" in dataset_loaders
    assert "huggingface" in dataset_loaders


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    file = tmp_path / "tasks.jsonl"
    write_jsonl(
        file,
        [
            {"id": "t1", "input": "2+2?", "target": "4", "topic": "math"},
            {"input": "capital of France?", "target": "Paris"},
        ],
    )
    spec = DatasetSpec(name="tasks", source="jsonl", location=str(file))
    samples = list(load_dataset(spec))
    assert [s.id for s in samples] == ["t1", "tasks-2"]
    assert samples[0].metadata == {"topic": "math"}
    assert samples[1].target == "Paris"


def test_jsonl_missing_input_rejected(tmp_path: Path) -> None:
    file = tmp_path / "bad.jsonl"
    write_jsonl(file, [{"target": "orphan"}])
    spec = DatasetSpec(name="bad", source="jsonl", location=str(file))
    with pytest.raises(ValueError, match="input"):
        list(load_dataset(spec))


def test_jsonl_missing_file_rejected(tmp_path: Path) -> None:
    spec = DatasetSpec(name="ghost", source="jsonl", location=str(tmp_path / "ghost.jsonl"))
    with pytest.raises(FileNotFoundError):
        list(load_dataset(spec))


def test_blank_lines_skipped(tmp_path: Path) -> None:
    file = tmp_path / "gaps.jsonl"
    file.write_text('{"input": "a"}\n\n{"input": "b"}\n', encoding="utf-8")
    spec = DatasetSpec(name="gaps", source="jsonl", location=str(file))
    assert len(list(load_dataset(spec))) == 2


def test_spec_from_yaml(tmp_path: Path) -> None:
    file = tmp_path / "data.yaml"
    file.write_text(
        "name: demo\nsource: jsonl\nlocation: demo.jsonl\nsplit: train\n", encoding="utf-8"
    )
    spec = DatasetSpec.from_yaml(file)
    assert spec.name == "demo"
    assert spec.source == "jsonl"
