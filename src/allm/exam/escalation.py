"""Heuristics for when to escalate grading to an LLM judge."""

from __future__ import annotations

import re

from allm.exam.text import normalise

_STOP = frozenset({"the", "a", "an", "is", "it", "of", "to", "in", "and", "that", "this", "are"})
_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}


def _content_words(text: str) -> set[str]:
    return {word for word in normalise(text).split() if word not in _STOP}


def might_be_paraphrase(answer: str, expected: str | None) -> bool:
    """Return True when a failed exact-match might still be correct.

    Skips escalation for clearly unrelated answers (saves judge calls).
    """
    if expected is None:
        return True
    answer_norm = normalise(answer)
    expected_norm = normalise(expected)
    if not answer_norm:
        return False

    answer_words = _content_words(answer)
    expected_words = _content_words(expected)
    if answer_words & expected_words:
        return True

    for ew in expected_words:
        for aw in answer_words:
            if len(ew) >= 4 and (ew in aw or aw in ew):
                return True

    if re.fullmatch(r"\d+", expected_norm):
        answer_digits = re.findall(r"\d+", answer_norm)
        if answer_digits:
            return expected_norm in answer_digits
        expected_word = _DIGIT_WORDS.get(expected_norm)
        if expected_word and expected_word in answer_words:
            return True
        return len(answer_norm.split()) >= 2 and bool(re.search(r"[a-z]", answer_norm))

    if len(expected_norm) <= 12 and len(answer_norm.split()) >= 2:
        return True

    return False
