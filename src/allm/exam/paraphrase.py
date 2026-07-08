"""Rule-based paraphrases for definition-style exam prompts."""

from __future__ import annotations

import re

_WHAT_IS = re.compile(r"^What is (.+)\?$", re.IGNORECASE)
_WHAT_ARE = re.compile(r"^What are (.+)\?$", re.IGNORECASE)
_WHAT_MEANS = re.compile(r"^What does (.+) mean\?$", re.IGNORECASE)
_WHAT_SHOULD = re.compile(
    r"^What should (.+) be like, according to Mr Keshe\?$",
    re.IGNORECASE,
)
_WE_CALL = re.compile(r"^What do the Kids workshops call (.+)\?$", re.IGNORECASE)
_IN_KIDS = re.compile(r"^In Kids plasma science, what is (.+)\?$", re.IGNORECASE)


def paraphrase_definition_prompt(prompt: str, *, variant: int = 0) -> str:
    """Return a semantically equivalent exam prompt (different wording)."""
    text = prompt.strip()
    if match := _WHAT_IS.match(text):
        subject = match.group(1)
        options = (
            f"Define {subject}.",
            f"Explain what {subject} means in the Kids plasma workshops.",
            f"According to Mr Keshe, what is {subject}?",
        )
        return options[variant % len(options)]
    if match := _WHAT_ARE.match(text):
        subject = match.group(1)
        options = (
            f"Define {subject}.",
            f"Explain what {subject} are in Kids plasma science.",
            f"How does Mr Keshe describe {subject}?",
        )
        return options[variant % len(options)]
    if match := _WHAT_MEANS.match(text):
        subject = match.group(1)
        options = (
            f"Define {subject}.",
            f"What is the meaning of {subject} in the Kids workshops?",
        )
        return options[variant % len(options)]
    if match := _WHAT_SHOULD.match(text):
        subject = match.group(1)
        options = (
            f"How should {subject} be described, per Mr Keshe?",
            f"What should {subject} be like in plasma teaching?",
        )
        return options[variant % len(options)]
    if match := _WE_CALL.match(text):
        subject = match.group(1)
        options = (
            f"What name do the Kids workshops give to {subject}?",
            f"In Kids plasma science, what are {subject} called?",
        )
        return options[variant % len(options)]
    if match := _IN_KIDS.match(text):
        subject = match.group(1)
        options = (
            f"Define {subject} as taught in the Kids workshops.",
            f"What is {subject} according to Mr Keshe?",
        )
        return options[variant % len(options)]
    return text
