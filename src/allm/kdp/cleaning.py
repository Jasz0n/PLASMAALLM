"""Stage 2 — Cleaning.

Removes surface noise (filler speech, stutter repetitions, formatting
junk) without touching semantic content. Cleaning is paragraph-wise:
each cleaned segment keeps the raw span of the paragraph it came from,
so provenance survives even though characters were removed.

The filler list is conservative on purpose: words like "like" or
"well" carry meaning too often to strip safely — losing semantics
violates KDP.md 7.1, keeping some noise does not.
"""

from __future__ import annotations

import re

from allm.kdp.types import CleanSegment, Document, SpanRef

# Standalone fillers, only removed as whole words/phrases.
_FILLERS = re.compile(
    r"\b(?:um+|uh+|er+|ah+|hmm+|you know|i mean|kind of like|sort of like)\b[,\s]*",
    re.IGNORECASE,
)
# Immediate word stutters: "the the", "we we we".
_STUTTER = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
_WHITESPACE = re.compile(r"[ \t]+")
_PARAGRAPH = re.compile(r"\n\s*\n")
# ASR pause markers (video silence → "...") and stage directions.
_ASR_ELLIPSIS = re.compile(r"\.{2,}")
_STAGE_DIRECTION = re.compile(
    r"\(\s*(?:"
    r"chuckle[s]?|laugh(?:ter|ing)?|inaudible|background sound"
    r")\s*\)",
    re.IGNORECASE,
)
_TRAILING_FRAGMENT = re.compile(
    r"\b(?:but|so|and|or|yeah|well|then),?\s*$",
    re.IGNORECASE,
)


def clean_asr_artifacts(text: str) -> str:
    """Remove transcription noise: pause ellipses, stage directions, comma junk."""
    text = _STAGE_DIRECTION.sub("", text)
    text = _ASR_ELLIPSIS.sub(" ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r",\s*,", ", ", text)
    text = _FILLERS.sub("", text)
    text = _STUTTER.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([?!])\.", r"\1", text)  # "Rick?." → "Rick?"
    text = re.sub(r",\s*\.", ".", text)  # "it's,." → "it's."
    text = re.sub(r"\bwell\s+", "", text, flags=re.IGNORECASE)  # pause filler after ellipsis removal
    text = _TRAILING_FRAGMENT.sub("", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Clean one string (exposed for tests and reuse)."""
    text = _FILLERS.sub("", text)
    text = _STUTTER.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # space before punctuation
    return text.strip()


def clean_document(document: Document) -> list[CleanSegment]:
    """Split into paragraphs, clean each, keep raw spans.

    Empty paragraphs (pure formatting noise) are dropped; that removes
    layout, never meaning.
    """
    segments: list[CleanSegment] = []
    cursor = 0
    for raw_paragraph in _PARAGRAPH.split(document.text):
        start = document.text.index(raw_paragraph, cursor)
        end = start + len(raw_paragraph)
        cursor = end
        cleaned = clean_text(raw_paragraph)
        if not cleaned:
            continue
        segments.append(
            CleanSegment(
                doc_id=document.id,
                text=cleaned,
                span=SpanRef(doc=document.id, start=start, end=end),
                context=document.context,
            )
        )
    return segments
