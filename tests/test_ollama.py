"""Tests for Ollama model loader (HTTP mocked — no daemon required)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from allm.models.base import ModelSpec, load_model, model_loaders
from allm.models.ollama import OllamaModel, resolve_api_key, resolve_base_url


def ollama_spec(**overrides) -> ModelSpec:
    return ModelSpec(
        name="test-ollama",
        provider="ollama",
        model_id="tiny:latest",
        **overrides,
    )


def test_ollama_provider_registered() -> None:
    assert "ollama" in model_loaders


def test_resolve_base_url_local_default() -> None:
    spec = ollama_spec()
    assert resolve_base_url(spec) == "http://127.0.0.1:11434"


def test_resolve_base_url_from_spec() -> None:
    spec = ollama_spec(base_url="http://custom:11434/")
    assert resolve_base_url(spec) == "http://custom:11434"


def test_resolve_base_url_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://cloud.example")
    spec = ollama_spec(device="cloud")
    assert resolve_base_url(spec) == "https://cloud.example"


def test_resolve_api_key_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-token")
    spec = ollama_spec(device="cloud")
    assert resolve_api_key(spec) == "secret-token"


@patch("allm.models.ollama.urllib.request.urlopen")
def test_generate_parses_response(mock_urlopen: MagicMock) -> None:
    payload = json.dumps({"response": "Paris\nCONFIDENCE: 0.8"}).encode()
    mock_urlopen.return_value.__enter__.return_value.read.return_value = payload
    model = OllamaModel(ollama_spec(), "http://127.0.0.1:11434", None)
    text = model.generate("Capital of France?")
    assert "Paris" in text
    request = mock_urlopen.call_args[0][0]
    assert request.full_url.endswith("/api/generate")
    body = json.loads(request.data.decode())
    assert body["model"] == "tiny:latest"
    assert body["stream"] is False


@patch("allm.models.ollama.urllib.request.urlopen")
def test_generate_with_logprobs(mock_urlopen: MagicMock) -> None:
    payload = json.dumps(
        {
            "response": "Paris",
            "logprobs": [{"token": "Paris", "logprob": -0.2}],
        }
    ).encode()
    mock_urlopen.return_value.__enter__.return_value.read.return_value = payload
    model = OllamaModel(ollama_spec(), "http://127.0.0.1:11434", None)
    text, lps = model.generate_with_logprobs("Capital?")
    assert text == "Paris"
    assert lps == (-0.2,)


@patch("allm.models.ollama.urllib.request.urlopen")
def test_load_via_registry(mock_urlopen: MagicMock) -> None:
    payload = json.dumps({"response": "4"}).encode()
    mock_urlopen.return_value.__enter__.return_value.read.return_value = payload
    model = load_model(ollama_spec())
    assert model.generate("2+2?") == "4"
