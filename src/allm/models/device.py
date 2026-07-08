"""Device and dtype resolution for Hugging Face backends.

Centralises auto-detection so ``device: auto`` picks CUDA, then MPS
(Apple Silicon), then CPU with a clear log line — graceful degradation
without touching Ollama (which manages its own runtime).
"""

from __future__ import annotations

from allm.core.logging import get_logger

logger = get_logger("models.device")


def resolve_torch_device(requested: str) -> str:
    """Map ``auto`` / ``cuda`` / ``mps`` / ``cpu`` to an available device."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "device resolution needs the ML extras: pip install -e '.[ml]'"
        ) from exc

    if requested == "auto":
        if torch.cuda.is_available():
            chosen = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            chosen = "mps"
        else:
            chosen = "cpu"
        logger.info("device auto -> %s", chosen)
        return chosen

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu")
        return "cpu"
    if requested == "mps":
        mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not mps_ok:
            logger.warning("mps requested but unavailable; falling back to cpu")
            return "cpu"
    return requested


def resolve_torch_dtype(requested: str, device: str):
    """Resolve dtype string; prefer float16 on GPU, float32 on CPU/MPS."""
    import torch

    if requested != "auto":
        return getattr(torch, requested) if isinstance(requested, str) else requested
    if device == "cuda":
        return torch.float16
    return torch.float32
