"""Tests for allm.core.config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from allm.core.config import ALLMConfig, load_config


def test_defaults() -> None:
    config = load_config(environ={})
    assert config.logging.level == "INFO"
    assert config.storage.backend == "sqlite"


def test_yaml_file_overrides_defaults(tmp_path: Path) -> None:
    file = tmp_path / "allm.yaml"
    file.write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
    config = load_config(file, environ={})
    assert config.logging.level == "DEBUG"
    assert config.storage.backend == "sqlite"  # untouched default


def test_env_overrides_yaml(tmp_path: Path) -> None:
    file = tmp_path / "allm.yaml"
    file.write_text("logging:\n  level: DEBUG\n", encoding="utf-8")
    config = load_config(file, environ={"ALLM_LOGGING__LEVEL": "WARNING"})
    assert config.logging.level == "WARNING"


def test_explicit_overrides_win(tmp_path: Path) -> None:
    config = load_config(
        environ={"ALLM_LOGGING__LEVEL": "WARNING"},
        overrides={"logging": {"level": "ERROR"}},
    )
    assert config.logging.level == "ERROR"


def test_resolved_anchors_relative_paths(tmp_path: Path) -> None:
    config = ALLMConfig(project_root=tmp_path).resolved()
    assert config.storage.path.is_absolute()
    assert config.storage.path == tmp_path.resolve() / "experiments/allm.sqlite3"


def test_config_is_immutable() -> None:
    config = ALLMConfig()
    with pytest.raises(ValidationError):
        config.project_root = Path("/elsewhere")  # type: ignore[misc]


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    file = tmp_path / "bad.yaml"
    file.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(file, environ={})
