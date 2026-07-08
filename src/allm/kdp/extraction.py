"""Stage 4 — Concept Extraction.

Rule-based, deterministic extractors over segments:

    definition     "X is/are/means/refers to ..."        -> type "concept"
    misconception  marker phrases ("common mistake", ...) -> "misconception"
    procedure      step markers (numbered, first/then)    -> "procedure"
    question       sentence ending in "?"                 -> "question"
    fact           generalisations ("always", numbers...) -> "fact"
    example        "for example" / "e.g."                 -> "fact" + tag

A segment matching nothing produces nothing — that is the noise filter,
not a loss of meaning: the raw document is still fully preserved in
Stage 1. Extraction quality improves by adding patterns here, never by
loosening provenance.
"""

from __future__ import annotations

import re

from allm.kdp.concept_quality import is_noisy_concept
from allm.kdp.types import RawKnowledgeUnit, Segment

_DEFINITION = re.compile(
    r"(?P<subject>[A-Z][A-Za-z0-9 \-]{1,60}?)\s+"
    r"(?:is|are|means|refers to|is defined as|is called)\s+"
    r"(?P<body>[^.!?]{3,})",
)
_MISCONCEPTION = re.compile(
    r"common (?:mistake|misconception)|people often (?:think|believe|assume)"
    r"|contrary to popular belief|it is (?:a myth|wrong to think)|many believe",
    re.IGNORECASE,
)
_PROCEDURE = re.compile(
    r"^\s*\d+[.)]\s|\b(?:first|then|next|after that|finally)\b[, ]", re.IGNORECASE | re.MULTILINE
)
_FACT = re.compile(r"\b(?:always|never|every|all|typically|consists of|\d+(?:\.\d+)?%?)\b")
_EXAMPLE = re.compile(r"\bfor example\b|\be\.g\.", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _concept_hint(text: str) -> str:
    """Best-effort subject phrase: the definition subject if present,
    otherwise the first few content-bearing words of the segment."""
    match = _DEFINITION.search(text)
    if match:
        return match["subject"].strip()
    words = re.findall(r"[A-Za-z][\w\-]*", text)
    return " ".join(words[:4]) if words else "unknown"


def _usable_concept(name: str) -> bool:
    """Reject ASR fragment labels that would pollute the knowledge graph."""
    stripped = name.strip()
    return bool(stripped) and stripped.lower() != "unknown" and not is_noisy_concept(stripped)


def _misconception_concept(text: str) -> str:
    """Prefer the belief topic over a filler-led segment hint."""
    match = re.search(
        r"people often (?:think|believe|assume)\s+(?P<topic>[^.!?]{3,80})",
        text,
        re.IGNORECASE,
    )
    if match:
        words = re.findall(r"[A-Za-z][\w\-]*", match.group("topic"))
        if words:
            topic = " ".join(words[:4])
            if _usable_concept(topic):
                return topic
    hint = _concept_hint(text)
    return hint if _usable_concept(hint) else "Common Misconception"


def extract(segments: list[Segment]) -> list[RawKnowledgeUnit]:
    """Run every extractor over every segment."""
    units: list[RawKnowledgeUnit] = []
    for seg in segments:
        units.extend(_extract_segment(seg))
    return units


def _extract_segment(seg: Segment) -> list[RawKnowledgeUnit]:
    units: list[RawKnowledgeUnit] = []
    common = {"span": seg.span, "context": seg.context}

    for match in _DEFINITION.finditer(seg.text):
        concept = match["subject"].strip()
        if not _usable_concept(concept):
            continue
        units.append(
            RawKnowledgeUnit(
                type="concept",
                concept=concept,
                content=match.group(0).strip(),
                **common,
            )
        )
    if _MISCONCEPTION.search(seg.text):
        units.append(
            RawKnowledgeUnit(
                type="misconception",
                concept=_misconception_concept(seg.text),
                content=seg.text,
                **common,
            )
        )
    if len(_PROCEDURE.findall(seg.text)) >= 2:
        concept = _concept_hint(seg.text)
        if _usable_concept(concept):
            units.append(
                RawKnowledgeUnit(
                    type="procedure",
                    concept=concept,
                    content=seg.text,
                    **common,
                )
            )
    for sentence in _SENTENCE.split(seg.text):
        sentence = sentence.strip()
        if sentence.endswith("?"):
            concept = _concept_hint(sentence)
            if not _usable_concept(concept):
                continue
            units.append(
                RawKnowledgeUnit(
                    type="question",
                    concept=concept,
                    content=sentence,
                    **common,
                )
            )
        elif _FACT.search(sentence) and not _DEFINITION.search(sentence):
            concept = _concept_hint(seg.text)
            if not _usable_concept(concept):
                continue
            tags = ("example",) if _EXAMPLE.search(sentence) else ()
            units.append(
                RawKnowledgeUnit(
                    type="fact",
                    concept=concept,
                    content=sentence,
                    tags=tags,
                    **common,
                )
            )
    return units
