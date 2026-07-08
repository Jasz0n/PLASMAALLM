"""Heuristics for reviewing KDP concept naming quality."""

from __future__ import annotations

import re

_NOISE_PREFIX = re.compile(
    r"^(and|or|but|so|if|when|as|in|on|at|to|this|that|there|then|now)\b",
    re.IGNORECASE,
)
_FRAGMENT = re.compile(r"^(one|two|all|any|some|many|each|every)\s", re.IGNORECASE)


def is_noisy_concept(name: str) -> bool:
    """Return True when a concept label looks like an ASR fragment, not a topic."""
    stripped = name.strip()
    if not stripped:
        return True
    if len(stripped.split()) > 9:
        return True
    if _NOISE_PREFIX.match(stripped):
        return True
    if _FRAGMENT.match(stripped) and len(stripped.split()) >= 4:
        return True
    if stripped.endswith((" Which", " You", " We", " They", " It", " This", " That")):
        return True
    return False


def concept_quality_report(names: list[str]) -> dict[str, int | float]:
    """Summarise how many concept labels pass the noise heuristic."""
    total = len(names)
    noisy = sum(1 for name in names if is_noisy_concept(name))
    clean = total - noisy
    ratio = (clean / total) if total else 1.0
    return {"total": total, "clean": clean, "noisy": noisy, "clean_ratio": ratio}
