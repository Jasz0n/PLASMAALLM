"""Ollama API model loader (local daemon or Ollama Cloud).

Uses the ``/api/generate`` endpoint with ``stream=false`` so the
interface matches other :class:`LanguageModel` backends. ``base_url``
and ``api_key`` come from the spec or environment (``OLLAMA_BASE_URL``,
``OLLAMA_CLOUD_BASE_URL``, ``OLLAMA_API_KEY``).

When ``logprobs=true`` is supported, :class:`OllamaModel` also satisfies
:class:`LogProbModel`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from allm.core.logging import get_logger
from allm.models.base import GenerationParams, ModelSpec, model_loaders

logger = get_logger("models.ollama")

_DEFAULT_BASE_URL = "http://127.0.0.1:11434"


def resolve_base_url(spec: ModelSpec) -> str:
    """Pick the Ollama host from spec or environment."""
    if spec.base_url:
        return spec.base_url.rstrip("/")
    if spec.device == "cloud":
        return os.environ.get("OLLAMA_CLOUD_BASE_URL", "https://ollama.com").rstrip("/")
    return os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def resolve_api_key(spec: ModelSpec) -> str | None:
    """Return bearer token when calling Ollama Cloud."""
    if spec.api_key:
        return spec.api_key
    if spec.device == "cloud" or resolve_base_url(spec).endswith("ollama.com"):
        return os.environ.get("OLLAMA_API_KEY")
    return None


def _extract_logprobs(data: dict[str, Any]) -> tuple[float, ...]:
    entries = data.get("logprobs") or []
    return tuple(float(entry["logprob"]) for entry in entries if "logprob" in entry)


class OllamaModel:
    """Wraps Ollama's generate API behind the LanguageModel protocol."""

    def __init__(self, spec: ModelSpec, base_url: str, api_key: str | None) -> None:
        self._spec = spec
        self._base_url = base_url
        self._api_key = api_key

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def _request(
        self,
        prompt: str,
        params: GenerationParams | None,
        *,
        logprobs: bool,
    ) -> dict[str, Any]:
        p = params or self._spec.generation
        payload: dict[str, Any] = {
            "model": self._spec.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": p.temperature,
                "top_p": p.top_p,
                "num_predict": p.max_new_tokens,
            },
        }
        if logprobs:
            payload["logprobs"] = True
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ollama generate failed ({exc.code}): {detail[:200]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"ollama unreachable at {self._base_url}: {exc.reason}"
            ) from exc

    def generate(self, prompt: str, params: GenerationParams | None = None) -> str:
        data = self._request(prompt, params, logprobs=False)
        text = data.get("response", "")
        if not text:
            raise RuntimeError(f"ollama returned empty response for {self._spec.model_id!r}")
        return text

    def generate_with_logprobs(
        self, prompt: str, params: GenerationParams | None = None
    ) -> tuple[str, tuple[float, ...]]:
        """Return completion text and per-token log probabilities."""
        data = self._request(prompt, params, logprobs=True)
        text = data.get("response", "")
        if not text:
            raise RuntimeError(f"ollama returned empty response for {self._spec.model_id!r}")
        return text, _extract_logprobs(data)


@model_loaders.register("ollama")
class OllamaModelLoader:
    """Loads models from a running Ollama daemon or Ollama Cloud."""

    def load(self, spec: ModelSpec) -> OllamaModel:
        base_url = resolve_base_url(spec)
        api_key = resolve_api_key(spec)
        logger.info("loading %s via ollama at %s", spec.model_id, base_url)
        return OllamaModel(spec, base_url, api_key)
