"""Configuration system.

Design decisions
----------------
- Pydantic models give us validation, defaults and IDE support for free.
- Configuration is layered: built-in defaults < YAML file < environment
  variables (``ALLM_`` prefix) < explicit overrides. Later layers win.
- Config objects are immutable (``frozen=True``) so components cannot
  mutate shared configuration at runtime — no hidden state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

ENV_PREFIX = "ALLM_"


class LoggingConfig(BaseModel):
    """How ALLM emits logs."""

    model_config = ConfigDict(frozen=True)

    level: str = "INFO"
    json_format: bool = False


class StorageConfig(BaseModel):
    """Where the versioned record store keeps its data."""

    model_config = ConfigDict(frozen=True)

    backend: str = "sqlite"
    path: Path = Path("experiments/allm.sqlite3")


class TrackingConfig(BaseModel):
    """Where experiment runs are recorded."""

    model_config = ConfigDict(frozen=True)

    backend: str = "local"
    root: Path = Path("experiments/runs")


class ALLMConfig(BaseModel):
    """Top-level project configuration.

    All paths are interpreted relative to ``project_root`` unless they
    are absolute; :meth:`resolved` returns a copy with absolute paths.
    """

    model_config = ConfigDict(frozen=True)

    project_root: Path = Path(".")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    def resolved(self) -> "ALLMConfig":
        """Return a copy where every relative path is anchored at project_root."""
        root = self.project_root.resolve()

        def anchor(p: Path) -> Path:
            return p if p.is_absolute() else root / p

        return self.model_copy(
            update={
                "project_root": root,
                "storage": self.storage.model_copy(update={"path": anchor(self.storage.path)}),
                "tracking": self.tracking.model_copy(update={"root": anchor(self.tracking.root)}),
            }
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base`` recursively; override wins."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """Translate ``ALLM_SECTION__FIELD=value`` variables into a nested dict.

    Example: ``ALLM_LOGGING__LEVEL=DEBUG`` -> ``{"logging": {"level": "DEBUG"}}``.
    """
    result: dict[str, Any] = {}
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = name[len(ENV_PREFIX) :].lower().split("__")
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return result


def load_config(
    path: Path | str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> ALLMConfig:
    """Build an :class:`ALLMConfig` from defaults, a YAML file, env vars
    and explicit overrides (in that order of precedence).

    Args:
        path: Optional YAML file. Missing file is an error; ``None`` means
            "defaults only".
        overrides: Highest-precedence values, e.g. from CLI flags.
        environ: Environment mapping (defaults to ``os.environ``);
            injectable for tests.
    """
    data: dict[str, Any] = {}
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping")
        data = _deep_merge(data, loaded)
    data = _deep_merge(data, _env_overrides(dict(environ if environ is not None else os.environ)))
    if overrides:
        data = _deep_merge(data, overrides)
    return ALLMConfig.model_validate(data)
