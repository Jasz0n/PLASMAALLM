"""Tests for allm.kdp: every stage plus end-to-end determinism."""

from pathlib import Path

import pytest

from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kdp.cleaning import clean_asr_artifacts, clean_document, clean_text
from allm.kdp.extraction import extract
from allm.kdp.normalization import canonical_tokens, normalize
from allm.kdp.segmentation import segment
from allm.kdp.types import RawKnowledgeUnit, SpanRef
from allm.knowledge import KnowledgeGraph
from allm.storage import SQLiteRecordStore

TRANSCRIPT_A = """Self-attention is a mechanism that relates tokens to each other.
It computes weights between every pair of tokens. For example, 100% of pairs are scored.

Um, so, people often think attention looks at words one by one, but it does not.

How does self attention scale with sequence length?
"""

TRANSCRIPT_B = """The self attention mechanism is a lookup over all token pairs.

To compute it: first project the inputs, then compare queries with keys, finally weight the values.
"""


def distill(*texts: str, aliases: dict[str, str] | None = None):
    store = DocumentStore()
    for i, text in enumerate(texts):
        store.ingest_text(f"t{i}.md", text, context="transformers")
    return KDPipeline(aliases=aliases).distill(store)


# -- Stage 1: ingestion --------------------------------------------------


def test_ingestion_ids_are_content_derived(tmp_path: Path) -> None:
    store = DocumentStore()
    a = store.ingest_text("a.md", "hello")
    assert a.id == store.ingest_text("a.md", "hello").id  # idempotent
    assert a.id != store.ingest_text("a.md", "different").id


def test_ingestion_preserves_raw_text_versioned(tmp_path: Path) -> None:
    record_store = SQLiteRecordStore(tmp_path / "kdp.sqlite3")
    store = DocumentStore(record_store)
    doc = store.ingest_text("a.md", "raw  text   with mess", context="x")
    stored = record_store.get("documents", doc.id)
    assert stored.value["text"] == "raw  text   with mess"  # untouched
    record_store.close()


# -- Stage 2: cleaning ---------------------------------------------------


def test_cleaning_removes_noise_keeps_meaning() -> None:
    assert clean_text("Um, the the model, you know, learns weights.") == (
        "the model, learns weights."
    )


def test_asr_artifacts_remove_pause_ellipses() -> None:
    raw = "I learned... because, Mr Keshe said something... real, really important."
    assert "..." not in clean_asr_artifacts(raw)
    assert "I learned because" in clean_asr_artifacts(raw)


def test_asr_artifacts_remove_stage_directions() -> None:
    text = clean_asr_artifacts("KD: Oh! Hi Mr Keshe. (chuckles) MK: What are the questions?")
    assert "(chuckles)" not in text
    assert "What are the questions?" in text


def test_cleaning_spans_reference_raw_document() -> None:
    store = DocumentStore()
    doc = store.ingest_text("a.md", "First paragraph.\n\nSecond um paragraph.")
    segments = clean_document(doc)
    assert len(segments) == 2
    raw = doc.text[segments[1].span.start : segments[1].span.end]
    assert raw == "Second um paragraph."  # span points at the *raw* text


# -- Stage 3: segmentation -------------------------------------------------


def test_topic_shift_starts_new_segment() -> None:
    store = DocumentStore()
    doc = store.ingest_text(
        "a.md",
        "Gradients flow backwards through the network. The network gradients vanish. "
        "Bananas are yellow tropical fruit.",
    )
    segments = segment(clean_document(doc))
    assert len(segments) == 2
    assert "Bananas" in segments[1].text


# -- Stage 4: extraction -----------------------------------------------------


def test_extractors_type_units() -> None:
    result = distill(TRANSCRIPT_A)
    kinds = {u.type for u in result.units}
    assert {"concept", "misconception", "question", "fact"} <= kinds


def test_noisy_concept_fragments_are_filtered() -> None:
    noisy = "And That is something we talk about in the workshop."
    result = distill(noisy)
    assert not any(u.normalized_concept == "And That" for u in result.units)


# -- Stage 5: normalization ---------------------------------------------------


def test_canonical_tokens_depluralise_and_drop_stopwords() -> None:
    assert canonical_tokens("the self-attention mechanisms") == ("self", "attention")


def test_variant_phrasings_merge() -> None:
    result = distill(TRANSCRIPT_A, TRANSCRIPT_B)
    concepts = {u.normalized_concept for u in result.units if u.type == "concept"}
    assert len([c for c in concepts if "attention" in c.lower()]) == 1


def test_alias_table_wins() -> None:
    units = [
        RawKnowledgeUnit(
            type="concept",
            concept="QKV trick",
            content="QKV trick is projection",
            span=SpanRef(doc="d", start=0, end=10),
            context="x",
        )
    ]
    normalized = normalize(units, aliases={"qkv trick": "Self-Attention"})
    assert normalized[0].concept == "Self-Attention"


# -- Stages 6-8: merge, conflicts, confidence -----------------------------------


def test_dedup_merges_sources_and_keeps_perspectives() -> None:
    result = distill(TRANSCRIPT_A, TRANSCRIPT_B)
    unit = next(u for u in result.units if u.type == "concept" and "ttention" in u.normalized_concept)
    assert len(unit.sources) == 2
    assert "--- perspective ---" in unit.content  # both explanations kept
    assert len(unit.raw_span_refs) >= 2


def test_more_sources_higher_confidence() -> None:
    single = distill(TRANSCRIPT_A)
    double = distill(TRANSCRIPT_A, TRANSCRIPT_B)

    def conf(result):
        return next(
            u.confidence
            for u in result.units
            if u.type == "concept" and "ttention" in u.normalized_concept
        )

    assert conf(double) > conf(single)


def test_contradiction_produces_conflict_and_penalty() -> None:
    contradiction_a = "Dropout is a regularisation technique that removes random units."
    contradiction_b = "Dropout is the compression of gradients for faster syncing."
    result = distill(contradiction_a, contradiction_b)
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.concept == "Dropout"
    assert conflict.interpretation_a != conflict.interpretation_b
    unit = next(u for u in result.units if u.normalized_concept == "Dropout")
    assert unit.confidence < 0.5  # penalised


def test_numeric_contradiction_detected() -> None:
    a = "The Peltier Stack is a converter that exceeds 80 percent efficiency."
    b = "The Peltier Stack is a module capped near 10 percent efficiency."
    result = distill(a, b)
    assert len(result.conflicts) == 1
    assert result.conflicts[0].concept == "The Peltier Stack"


def test_agreeing_numbers_do_not_conflict() -> None:
    a = "The Peltier Stack is a module rated at 10 percent efficiency."
    b = "The Peltier Stack is a converter reaching 10 percent efficiency in tests."
    assert distill(a, b).conflicts == ()


def test_consistent_definitions_do_not_conflict() -> None:
    a = "Dropout is a regularisation technique that removes random units."
    b = "Dropout is a regularisation method removing units at random."
    assert distill(a, b).conflicts == ()


# -- determinism -----------------------------------------------------------------


def test_same_input_same_output_including_ids() -> None:
    first = distill(TRANSCRIPT_A, TRANSCRIPT_B)
    second = distill(TRANSCRIPT_A, TRANSCRIPT_B)
    assert first == second


# -- Stage 9: graph injection ------------------------------------------------------


@pytest.fixture()
def graph(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "kg.sqlite3")
    yield KnowledgeGraph(store), store
    store.close()


def test_injection_adds_then_revises(graph) -> None:
    kg, record_store = graph
    result = distill(TRANSCRIPT_A, TRANSCRIPT_B)
    injector = GraphInjector(kg, record_store)

    report = injector.inject(result)
    assert report["added"] > 0
    concept_name = next(
        u.normalized_concept for u in result.units if "ttention" in u.normalized_concept
    )
    before = kg.get(concept_name)
    assert before is not None
    assert before.source == "kdp"
    assert before.evidence

    # re-injection revises (appends evidence), never duplicates or deletes
    versions_before = len(kg.history(concept_name))
    report2 = injector.inject(result)
    assert report2["added"] == 0
    after = kg.get(concept_name)
    assert len(after.evidence) > len(before.evidence)
    assert len(kg.history(concept_name)) > versions_before


def test_conflicts_are_persisted_and_marked(graph) -> None:
    kg, record_store = graph
    result = distill(
        "Dropout is a regularisation technique that removes random units.",
        "Dropout is the compression of gradients for faster syncing.",
    )
    GraphInjector(kg, record_store).inject(result)
    stored = record_store.keys("conflicts")
    assert stored and stored[0].startswith("Dropout")
    concept = kg.get("Dropout")
    assert any(not e.supports for e in concept.evidence)  # disagreement visible
