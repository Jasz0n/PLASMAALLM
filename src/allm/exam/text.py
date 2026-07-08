"""Shared text normalisation for exams and grading."""

from __future__ import annotations

import re


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text.strip(" .,!?:;\"'")
