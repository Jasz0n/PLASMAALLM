"""Confidence extraction from model output.

Phase 3 uses *self-reported* confidence: the student is instructed to
end its answer with a ``CONFIDENCE: <0..1>`` line, which we parse and
clamp. This is the weakest honest signal (models are poorly
calibrated), but it establishes the interface; a log-prob based
estimator can be added later for backends that expose token
probabilities, without touching the students.
"""

from __future__ import annotations

import re

CONFIDENCE_INSTRUCTION = (
    "After your answer, on a new final line, write 'CONFIDENCE: x' "
    "where x is a number between 0 and 1 for how sure you are."
)

_PATTERN = re.compile(r"^\s*CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_confidence(text: str) -> tuple[str, float | None]:
    """Split model output into (answer, confidence).

    Returns the text with the confidence line removed, and the parsed
    value clamped to [0, 1] — or ``None`` if no line was found, so the
    caller can apply its own default rather than us inventing one here.
    """
    match = _PATTERN.search(text)
    if match is None:
        return text.strip(), None
    value = max(0.0, min(1.0, float(match.group(1))))
    answer = (text[: match.start()] + text[match.end() :]).strip()
    return answer, value
