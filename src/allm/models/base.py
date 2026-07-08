"""Model interfaces.

Design decisions
----------------
- :class:`ModelSpec` is a validated, serialisable description of a model
  (what to load); :class:`LanguageModel` is the behavioural interface
  (what a loaded model can do). Keeping them separate lets configs in
  ``configs/models/`` fully describe teacher/student models as data.
- ``generate`` is deliberately the only required capability for Phase 1.
  Later phases will extend the surface (log-probs for confidence,
  fine-tuning hooks) as separate protocols, so simple backends are not
  forced to implement everything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml
from pydantic import BaseModel, ConfigDict, Field

from allm.core.registry import Registry


class GenerationParams(BaseModel):
    """Decoding parameters with conservative defaults."""

    model_config = ConfigDict(frozen=True)

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


class ModelSpec(BaseModel):
    """Declarative description of a model, loadable from YAML.

    ``provider`` selects the loader from :data:`model_loaders`;
    ``model_id`` is interpreted by that loader (e.g. a Hugging Face hub
    id or a local path).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    provider: str = "huggingface"
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    base_url: str | None = None
    api_key: str | None = None
    generation: GenerationParams = Field(default_factory=GenerationParams)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ModelSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


@runtime_checkable
class LanguageModel(Protocol):
    """A loaded model that can complete text."""

    @property
    def spec(self) -> ModelSpec: ...

    def generate(self, prompt: str, params: GenerationParams | None = None) -> str:
        """Return the completion for ``prompt`` (not echoing the prompt)."""
        ...


@runtime_checkable
class LogProbModel(LanguageModel, Protocol):
    """A model that can return per-token log probabilities."""

    def generate_with_logprobs(
        self, prompt: str, params: GenerationParams | None = None
    ) -> tuple[str, tuple[float, ...]]:
        """Return ``(completion, token_logprobs)``."""
        ...


@runtime_checkable
class ModelLoader(Protocol):
    """Turns a :class:`ModelSpec` into a :class:`LanguageModel`."""

    def load(self, spec: ModelSpec) -> LanguageModel: ...


model_loaders: Registry[type] = Registry("model_loader")


def load_model(spec: ModelSpec) -> LanguageModel:
    """Convenience: pick the loader for ``spec.provider`` and load."""
    loader_cls = model_loaders.get(spec.provider)
    return loader_cls().load(spec)
