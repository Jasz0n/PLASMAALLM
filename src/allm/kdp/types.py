"""KDP value objects — the artifacts each pipeline stage produces.

Stage outputs (KDP.md section 4):

    Stage 1  Document        raw text, never modified
    Stage 2  CleanSegment    cleaned paragraph + raw span it came from
    Stage 3  Segment         one idea, with provenance
    Stage 4  RawKnowledgeUnit  typed extraction, still un-normalised
    Stage 5-8 KnowledgeUnit  normalised, merged, scored
    Stage 7  ConflictNode    preserved disagreement (first-class output)

Everything is frozen and JSON-serialisable; ids are content-derived so
the whole pipeline stays deterministic (same input -> same output,
including ids).
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

KnowledgeUnitType = Literal["concept", "fact", "procedure", "misconception", "question"]


def content_hash(*parts: str) -> str:
    """Deterministic 8-hex-char id component."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:8]


class SpanRef(BaseModel):
    """Reference back into a raw document (paragraph-granular)."""

    model_config = ConfigDict(frozen=True)

    doc: str
    start: int
    end: int


class Document(BaseModel):
    """One ingested source. ``text`` is the raw stream, never modified."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    text: str
    context: str = "general"


class CleanSegment(BaseModel):
    """A cleaned paragraph plus the raw span it was derived from."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    text: str
    span: SpanRef
    context: str


class Segment(BaseModel):
    """One semantic unit (Stage 3: one idea per segment)."""

    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    text: str
    span: SpanRef
    context: str


class RawKnowledgeUnit(BaseModel):
    """A typed extraction before normalisation and merging."""

    model_config = ConfigDict(frozen=True)

    type: KnowledgeUnitType
    concept: str
    content: str
    span: SpanRef
    context: str
    tags: tuple[str, ...] = ()


class KnowledgeUnit(BaseModel):
    """The finalised, machine-learnable unit (KDP.md section 3)."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: KnowledgeUnitType
    content: str
    normalized_concept: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: tuple[str, ...]
    context: str
    relations: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    raw_span_refs: tuple[SpanRef, ...]


class ConflictNode(BaseModel):
    """Preserved disagreement between sources — not an error."""

    model_config = ConfigDict(frozen=True)

    concept: str
    interpretation_a: str
    interpretation_b: str
    sources: tuple[str, ...]
    evidence: tuple[SpanRef, ...]
