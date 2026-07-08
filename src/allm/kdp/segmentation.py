"""Stage 3 — Segmentation.

Splits cleaned paragraphs into semantic units, not fixed lengths:
sentences are accumulated into the current segment while they share
vocabulary with it; a sentence with (almost) no lexical overlap with
the running segment signals a topic shift and starts a new one.
Paragraph boundaries always split.

Deterministic by construction: pure string operations, fixed threshold.
"""

from __future__ import annotations

import re

from allm.kdp.types import CleanSegment, Segment, content_hash

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9]+")

# Words too common to indicate shared topic.
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have if in into is it its of on or "
    "so than that the their then there these this to was we what when which will "
    "with you your not can may do does".split()
)

TOPIC_SHIFT_THRESHOLD = 0.10


def content_tokens(text: str) -> set[str]:
    """Lowercased tokens minus stopwords (shared by later stages)."""
    return {t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def segment(clean_segments: list[CleanSegment]) -> list[Segment]:
    """Split cleaned paragraphs into one-idea segments."""
    segments: list[Segment] = []
    for paragraph in clean_segments:
        sentences = _SENTENCE.split(paragraph.text)
        groups: list[list[str]] = []
        running: set[str] = set()
        for sentence in sentences:
            tokens = content_tokens(sentence)
            if groups and _overlap(running, tokens) <= TOPIC_SHIFT_THRESHOLD:
                groups.append([sentence])  # topic shift
                running = set(tokens)
            elif not groups:
                groups.append([sentence])
                running = set(tokens)
            else:
                groups[-1].append(sentence)
                running |= tokens
        for group in groups:
            text = " ".join(group).strip()
            if not text:
                continue
            segments.append(
                Segment(
                    id=f"seg_{content_hash(paragraph.doc_id, text)}",
                    doc_id=paragraph.doc_id,
                    text=text,
                    span=paragraph.span,
                    context=paragraph.context,
                )
            )
    return segments
