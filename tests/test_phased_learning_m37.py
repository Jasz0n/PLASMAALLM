"""Tests for M37 phased book/workshop learning."""

from __future__ import annotations

import pytest

from allm.data.base import Sample
from allm.loop.phased_learning import (
    LearningPhase,
    discovery_order_from_phase,
    parse_learning_phases,
)
from allm.researcher.capabilities.registry import pipeline_order
from allm.researcher.types import KnowledgePackage
from allm.teacher.source_training import (
    BOOK_PROVIDER,
    filter_workshop_samples,
    samples_from_book_packages,
)


def test_parse_learning_phases_books_then_workshops(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_KEL_PHASE_ORDER", "books_then_workshops")
    monkeypatch.setenv("ALLM_KEL_BOOK_ITERS", "3")
    monkeypatch.setenv("ALLM_KEL_WORKSHOP_ITERS", "2")
    phases = parse_learning_phases()
    assert phases == (
        LearningPhase(source="book", iterations=3),
        LearningPhase(source="workshop", iterations=2),
    )


def test_discovery_order_from_phase(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_KEL_PHASE_ORDER", "books_then_workshops")
    assert discovery_order_from_phase() == "books_first"
    monkeypatch.setenv("ALLM_KEL_PHASE_ORDER", "workshops_then_books")
    assert discovery_order_from_phase() == "workshops_first"


def test_parse_learning_phases_unknown_order(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_KEL_PHASE_ORDER", "invalid")
    with pytest.raises(ValueError, match="unknown ALLM_KEL_PHASE_ORDER"):
        parse_learning_phases()


def test_pipeline_order_books_first() -> None:
    order = pipeline_order(None, discovery_order="books_first")
    assert order.index("discovery.book") < order.index("discovery.workshop")


def test_samples_from_book_packages() -> None:
    package = KnowledgePackage(
        id="book1",
        provider=BOOK_PROVIDER,
        title="Book One",
        curriculum_topic="kids-plasma",
        definitions=(("plasma", "ionized gas"), ("field", "magnetic influence")),
    )
    workshop = KnowledgePackage(
        id="ws1",
        provider="kids-workshops",
        title="Workshop One",
        curriculum_topic="kids-plasma",
        definitions=(("workshop", "live demo"),),
    )
    rows = samples_from_book_packages((package, workshop), topic="kids-plasma")
    assert len(rows) == 2
    assert all(row.metadata.get("source") == "book" for row in rows)
    assert rows[0].input == "What is plasma?"


def test_filter_workshop_samples() -> None:
    rows = [
        Sample(id="w1", input="a", target="b", metadata={"source": "workshop"}),
        Sample(id="b1", input="c", target="d", metadata={"source": "book"}),
    ]
    filtered = filter_workshop_samples(rows)
    assert len(filtered) == 1
    assert filtered[0].id == "w1"


def test_parse_learning_phases_books_only(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_KEL_PHASE_ORDER", "books_only")
    monkeypatch.setenv("ALLM_KEL_BOOK_ITERS", "8")
    phases = parse_learning_phases()
    assert phases == (LearningPhase(source="book", iterations=8),)


def test_discovery_order_books_only(monkeypatch) -> None:
    monkeypatch.setenv("ALLM_KEL_PHASE_ORDER", "books_only")
    assert discovery_order_from_phase() == "books_first"


def test_parse_learning_phases_empty(monkeypatch) -> None:
    monkeypatch.delenv("ALLM_KEL_PHASE_ORDER", raising=False)
    assert parse_learning_phases() is None
