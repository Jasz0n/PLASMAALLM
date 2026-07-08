"""Tests for allm.tracking."""

import json
from pathlib import Path

import pytest

from allm.tracking import LocalTracker, tracker_backends


def test_run_lifecycle(tmp_path: Path) -> None:
    tracker = LocalTracker(tmp_path)
    run = tracker.start_run("baseline")
    run.log_params({"lr": 1e-4, "model": "echo"})
    run.log_metric("accuracy", 0.5, step=1)
    run.log_metric("accuracy", 0.75, step=2)
    run.log_artifact("exam.txt", "Q1: what is gravity?")
    run.finish()

    meta = json.loads((run.directory / "meta.json").read_text())
    assert meta["name"] == "baseline"
    assert meta["status"] == "completed"

    params = json.loads((run.directory / "params.json").read_text())
    assert params == {"lr": 1e-4, "model": "echo"}

    metrics = [
        json.loads(line) for line in (run.directory / "metrics.jsonl").read_text().splitlines()
    ]
    assert [m["value"] for m in metrics] == [0.5, 0.75]

    assert (run.directory / "artifacts" / "exam.txt").read_text() == "Q1: what is gravity?"


def test_finished_run_rejects_logging(tmp_path: Path) -> None:
    run = LocalTracker(tmp_path).start_run("done")
    run.finish()
    with pytest.raises(RuntimeError, match="finished"):
        run.log_metric("x", 1.0)


def test_list_runs_ordered(tmp_path: Path) -> None:
    tracker = LocalTracker(tmp_path)
    first = tracker.start_run("a")
    second = tracker.start_run("b")
    assert tracker.list_runs() == sorted([first.run_id, second.run_id])


def test_artifact_names_sanitised(tmp_path: Path) -> None:
    run = LocalTracker(tmp_path).start_run("safe")
    run.log_artifact("../escape.txt", "content")
    files = list((run.directory / "artifacts").iterdir())
    assert len(files) == 1
    assert files[0].parent == run.directory / "artifacts"


def test_registered_as_backend() -> None:
    assert tracker_backends.get("local") is LocalTracker
