"""Tests for the allm CLI (invoked in-process via main())."""

import json
from pathlib import Path

import pytest

import allm
from allm.cli.main import main


def test_info(capsys: pytest.CaptureFixture) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert allm.__version__ in out


def test_config_show_default(capsys: pytest.CaptureFixture) -> None:
    assert main(["config", "show"]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["logging"]["level"] == "INFO"


def test_plugins_lists_registries(capsys: pytest.CaptureFixture) -> None:
    assert main(["plugins"]) == 0
    out = capsys.readouterr().out
    assert "model_loader" in out
    assert "echo" in out
    assert "sqlite" in out


def test_model_validate(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    spec = tmp_path / "model.yaml"
    spec.write_text("name: m\nprovider: echo\nmodel_id: none\n", encoding="utf-8")
    assert main(["model", "validate", str(spec)]) == 0
    assert "echo" in capsys.readouterr().out


def test_model_validate_unknown_provider_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    spec = tmp_path / "model.yaml"
    spec.write_text("name: m\nprovider: nonexistent\nmodel_id: none\n", encoding="utf-8")
    assert main(["model", "validate", str(spec)]) == 1


def test_dataset_peek(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    data = tmp_path / "d.jsonl"
    data.write_text('{"input": "a"}\n{"input": "b"}\n{"input": "c"}\n', encoding="utf-8")
    spec = tmp_path / "d.yaml"
    spec.write_text(f"name: d\nsource: jsonl\nlocation: {data}\n", encoding="utf-8")
    assert main(["dataset", "peek", str(spec), "-n", "2"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2


def test_runs_empty(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    config = tmp_path / "allm.yaml"
    config.write_text(f"project_root: {tmp_path}\n", encoding="utf-8")
    assert main(["runs", "-c", str(config)]) == 0
    assert "no runs" in capsys.readouterr().out


def test_benchmark_output_creates_parent_dirs(tmp_path: Path) -> None:
    """Regression for issue #1: a finished report must never be lost to
    a missing output directory."""
    output = tmp_path / "reports" / "nested" / "report.json"
    code = main(
        [
            "--log-level", "WARNING",
            "benchmark",
            "--corpora", "fiction",
            "--iterations", "1",
            "--root", str(Path(__file__).resolve().parents[1]),
            "--output", str(output),
        ]
    )
    assert code == 0
    assert output.exists() and '"corpus": "fiction"' in output.read_text()
