"""Tests for book page concept hint extraction."""

from allm.researcher.book_evidence import concept_hints_from_page_text


def test_concept_hints_from_page_text_skips_stopwords() -> None:
    hints = concept_hints_from_page_text(
        "The plasmatic magnetic fields interact through gravitational forces on Earth."
    )
    assert "plasmatic" in hints
    assert "magnetic" in hints
    assert "book" not in hints
