"""Language-model interfaces and loaders.

Teachers and students (later phases) talk to models exclusively through
the :class:`~allm.models.base.LanguageModel` protocol, so the concrete
backend (Hugging Face, API-served, mock) is a config decision.
"""

from allm.models.base import GenerationParams, LanguageModel, LogProbModel, ModelLoader, ModelSpec, load_model, model_loaders
from allm.models.echo import EchoModel, EchoModelLoader

# Registers the "huggingface" loader; safe to import without torch installed.
from allm.models import huggingface as _huggingface  # noqa: F401
from allm.models import ollama as _ollama  # noqa: F401

__all__ = [
    "GenerationParams",
    "LanguageModel",
    "LogProbModel",
    "ModelLoader",
    "ModelSpec",
    "load_model",
    "model_loaders",
    "EchoModel",
    "EchoModelLoader",
]
