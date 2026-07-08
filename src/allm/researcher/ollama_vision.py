"""Ollama vision API client for frame captioning."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from allm.core.logging import get_logger
from allm.models.ollama import resolve_api_key, resolve_base_url
from allm.models.base import ModelSpec

logger = get_logger("researcher.ollama_vision")

DEFAULT_VISION_MODEL = "llava"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


def encode_image_base64(path: Path | str) -> str:
    """Read a JPEG/PNG and return base64 for Ollama chat."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def build_vision_prompt(
    *,
    transcript_excerpt: str,
    tags: tuple[str, ...] = (),
    concept_hints: tuple[str, ...] = (),
) -> str:
    """Prompt for workshop frame description."""
    tag_text = ", ".join(tags[:6])
    hint_text = ", ".join(concept_hints[:4])
    return (
        "Describe what is visible in this workshop video frame for a student. "
        "Focus on objects, colors, motion, and plasma or magnet demonstrations if present. "
        "Be concise (2-3 sentences).\n"
        f"Transcript context: {transcript_excerpt[:200]}\n"
        f"Tags: {tag_text or 'none'}\n"
        f"Concepts: {hint_text or 'none'}"
    )


def ollama_reachable(base_url: str | None = None) -> bool:
    """Return True when Ollama responds to a tags probe."""
    root = (base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    try:
        request = urllib.request.Request(f"{root}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


class OllamaVisionClient:
    """Call Ollama /api/chat with a base64 image."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_VISION_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 120,
    ) -> None:
        spec = ModelSpec(name="vision", provider="ollama", model_id=model_id)
        self._model_id = model_id
        self._base_url = (base_url or resolve_base_url(spec)).rstrip("/")
        self._api_key = api_key if api_key is not None else resolve_api_key(spec)
        self._timeout = timeout_sec

    def describe_image(self, image_path: Path | str, prompt: str) -> str:
        """Return a vision model caption for one image file."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"frame not found: {path}")
        payload = {
            "model": self._model_id,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encode_image_base64(path)],
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ollama vision failed ({exc.code}): {detail[:200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ollama unreachable at {self._base_url}: {exc.reason}") from exc

        message = data.get("message") or {}
        text = str(message.get("content", "")).strip()
        if not text:
            raise RuntimeError(f"ollama vision returned empty caption for {self._model_id!r}")
        return text
