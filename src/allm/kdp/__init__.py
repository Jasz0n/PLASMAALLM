"""KDP — Knowledge Distillation Pipeline (see KDP.md, project root).

A deterministic, staged compiler from raw human explanation streams
(transcripts, notes, converted PDFs) to atomic Knowledge Units, and the
only entry point for unstructured knowledge into the Phase 5 graph:

    DocumentStore -> KDPipeline.distill() -> GraphInjector.inject()

Not a model: pure rule-based transformations, same input -> same
output, ids included.
"""

from allm.kdp.corpus import export_samples_jsonl, ingest_cleaned_corpus, load_samples_jsonl, units_to_samples
from allm.kdp.holdout import sample_source, split_samples_holdout, workshop_number
from allm.kdp.mk_samples import filter_samples_by_kind, mk_corpus_to_samples, parse_sample_kinds
from allm.kdp.ingestion import DocumentStore
from allm.kdp.pipeline import DistillationResult, GraphInjector, KDPipeline
from allm.kdp.types import (
    ConflictNode,
    Document,
    KnowledgeUnit,
    RawKnowledgeUnit,
    Segment,
    SpanRef,
)

__all__ = [
    "DocumentStore",
    "KDPipeline",
    "GraphInjector",
    "DistillationResult",
    "ConflictNode",
    "Document",
    "KnowledgeUnit",
    "RawKnowledgeUnit",
    "Segment",
    "SpanRef",
    "ingest_cleaned_corpus",
    "units_to_samples",
    "export_samples_jsonl",
    "load_samples_jsonl",
    "mk_corpus_to_samples",
    "filter_samples_by_kind",
    "parse_sample_kinds",
    "sample_source",
    "split_samples_holdout",
    "workshop_number",
]
