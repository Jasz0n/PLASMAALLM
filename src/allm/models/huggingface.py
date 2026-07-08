"""Hugging Face Transformers model loader.

``torch``/``transformers`` are imported lazily inside :meth:`load`, so
this module is always importable (and the ``huggingface`` provider is
always registered) even in environments without the ML extras — the
error only surfaces if someone actually tries to load such a model.
Install with: ``pip install -e ".[ml]"``.
"""

from __future__ import annotations

from allm.core.logging import get_logger
from allm.models.base import GenerationParams, ModelSpec, model_loaders
from allm.models.device import resolve_torch_device, resolve_torch_dtype

logger = get_logger("models.huggingface")


class HFModel:
    """Wraps a ``transformers`` causal LM behind the LanguageModel protocol."""

    def __init__(self, spec: ModelSpec, tokenizer, model, device: str) -> None:
        self._spec = spec
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def generate(self, prompt: str, params: GenerationParams | None = None) -> str:
        p = params or self._spec.generation
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=p.max_new_tokens,
            do_sample=p.temperature > 0,
            temperature=p.temperature if p.temperature > 0 else None,
            top_p=p.top_p,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


@model_loaders.register("huggingface")
class HFModelLoader:
    """Loads causal LMs from the Hugging Face hub or a local path."""

    def load(self, spec: ModelSpec) -> HFModel:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "the 'huggingface' provider needs the ML extras: pip install -e '.[ml]'"
            ) from exc

        device = resolve_torch_device(spec.device)
        dtype = resolve_torch_dtype(spec.dtype, device)
        logger.info("loading %s (%s) on %s dtype=%s", spec.model_id, spec.name, device, dtype)

        tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        load_kwargs: dict = {"torch_dtype": dtype}
        if device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)
            model = model.to("cpu")
        elif device in ("cuda", "mps"):
            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                **load_kwargs,
                device_map={"": device},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                **load_kwargs,
                device_map=spec.device,
            )
        model.eval()
        return HFModel(spec, tokenizer, model, device)
