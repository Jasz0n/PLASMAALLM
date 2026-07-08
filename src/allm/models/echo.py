"""Deterministic in-memory model for tests and dry runs.

Not a placeholder: the whole learning loop (teacher, students, exams,
debate) must be executable without GPUs or downloads, both in CI and
when debugging orchestration logic. ``EchoModel`` answers from a
scripted mapping and falls back to echoing, which makes higher-level
behaviour fully deterministic and assertable.
"""

from __future__ import annotations

from allm.models.base import GenerationParams, ModelSpec, model_loaders


class EchoModel:
    """Returns scripted answers; echoes the prompt otherwise."""

    def __init__(self, spec: ModelSpec, responses: dict[str, str] | None = None) -> None:
        self._spec = spec
        self._responses = dict(responses or {})

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def script(self, prompt: str, response: str) -> None:
        """Register a canned response for an exact prompt."""
        self._responses[prompt] = response

    def generate(self, prompt: str, params: GenerationParams | None = None) -> str:
        if prompt in self._responses:
            return self._responses[prompt]
        limit = (params or self._spec.generation).max_new_tokens
        return f"echo: {prompt}"[:limit]


@model_loaders.register("echo")
class EchoModelLoader:
    """Loader so ``provider: echo`` works from YAML specs."""

    def load(self, spec: ModelSpec) -> EchoModel:
        return EchoModel(spec)
