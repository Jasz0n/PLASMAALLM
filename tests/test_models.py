"""Tests for allm.models (echo backend + spec plumbing; no downloads)."""

from pathlib import Path

import pytest

from allm.models import EchoModel, LanguageModel, ModelSpec, model_loaders
from allm.models.base import load_model


def spec(**overrides) -> ModelSpec:
    return ModelSpec(name="test", provider="echo", model_id="none", **overrides)


def test_providers_registered() -> None:
    assert "echo" in model_loaders
    assert "huggingface" in model_loaders  # registered even without torch
    assert "ollama" in model_loaders


def test_load_model_via_registry() -> None:
    model = load_model(spec())
    assert isinstance(model, EchoModel)
    assert isinstance(model, LanguageModel)  # satisfies the protocol
    assert model.spec.name == "test"


def test_echo_scripted_and_fallback() -> None:
    model = EchoModel(spec(), responses={"2+2?": "4"})
    assert model.generate("2+2?") == "4"
    assert model.generate("hello") == "echo: hello"


def test_spec_from_yaml(tmp_path: Path) -> None:
    file = tmp_path / "model.yaml"
    file.write_text(
        "name: yaml-model\nprovider: echo\nmodel_id: none\n"
        "generation:\n  max_new_tokens: 8\n",
        encoding="utf-8",
    )
    loaded = ModelSpec.from_yaml(file)
    assert loaded.name == "yaml-model"
    assert loaded.generation.max_new_tokens == 8


def test_generation_params_respected() -> None:
    model = load_model(spec())
    out = model.generate("a very long prompt indeed", params=None)
    assert len(out) <= model.spec.generation.max_new_tokens


def test_hf_loader_without_ml_extras_raises_clear_error() -> None:
    try:
        import transformers  # noqa: F401

        pytest.skip("transformers installed; error path not reachable")
    except ImportError:
        pass
    hf_spec = ModelSpec(name="hf", provider="huggingface", model_id="gpt2")
    with pytest.raises(ImportError, match=r"\[ml\]"):
        load_model(hf_spec)
