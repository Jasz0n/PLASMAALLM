"""Log-probability confidence estimation.

Backends that expose per-token log probabilities (Ollama with
``logprobs=true``) can supply a signal that is often better calibrated
than self-reported ``CONFIDENCE:`` lines. The estimator is deliberately
simple: geometric mean token probability, clamped to [0, 1].
"""

from __future__ import annotations

import math
from typing import Sequence


def estimate_from_logprobs(logprobs: Sequence[float]) -> float | None:
    """Map mean log-probability to a confidence in [0, 1].

    Returns ``None`` when there are no usable values.
    """
    if not logprobs:
        return None
    mean_lp = sum(logprobs) / len(logprobs)
    return max(0.0, min(1.0, math.exp(mean_lp)))
