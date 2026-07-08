"""Experiment-tracking interface.

Design decisions
----------------
- The interface is the smallest one every tracker (local files, MLflow,
  W&B, ...) can satisfy: start a run, log params/metrics/artifacts,
  finish. Backends register in ``tracker_backends`` and are selected
  via configuration.
- A :class:`Run` is a handle object rather than global mutable state,
  so several runs can be open at once (e.g. one per student).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from allm.core.registry import Registry


@runtime_checkable
class Run(Protocol):
    """Handle to one experiment run."""

    @property
    def run_id(self) -> str: ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Record hyper-parameters / configuration for this run."""
        ...

    def log_metric(self, name: str, value: float, *, step: int | None = None) -> None:
        """Record one scalar metric observation."""
        ...

    def log_artifact(self, name: str, content: str) -> None:
        """Store a text artifact (report, transcript, generated exam...)."""
        ...

    def finish(self, status: str = "completed") -> None:
        """Mark the run as finished; no further logging is allowed."""
        ...


@runtime_checkable
class ExperimentTracker(Protocol):
    """Creates and lists runs."""

    def start_run(self, name: str) -> Run: ...

    def list_runs(self) -> list[str]:
        """Return run ids, oldest first."""
        ...


tracker_backends: Registry[type] = Registry("tracker_backend")
