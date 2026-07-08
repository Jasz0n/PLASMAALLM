"""Tests for MK prose sample extraction."""

from pathlib import Path

from allm.kdp.mk_samples import (
    dedupe_samples,
    definition_prompt,
    is_good_subject,
    mk_corpus_to_samples,
    paragraph_to_samples,
    parse_sample_kinds,
)

PARAGRAPH = """
Plasma is the way the Universe works. Matter and physics is the way the man works.
In fact, a Plasma, to a child, should be an orange in rotation, but it rotates all the time.
Magnetic fields are not the way this is. Fields exist in the Universe in a round shape.
After a few years, they saw that metal bird, we call it an airplane, appeared in Tehran.
What are the questions?
"""


def test_rejects_bad_subjects() -> None:
    assert not is_good_subject("And That")
    assert not is_good_subject("Yeah")
    assert not is_good_subject("Withholding information until the time")
    assert is_good_subject("Plasma")
    assert is_good_subject("Magnetic fields")


def test_definition_prompt_short() -> None:
    assert definition_prompt("Plasma", "is") == "What is Plasma?"
    assert definition_prompt("Magnetic fields", "are") == "What are Magnetic fields?"


def test_paragraph_extracts_definitions_and_compact() -> None:
    samples = paragraph_to_samples(PARAGRAPH, "workshop3.txt")
    prompts = [s.input for s in samples]
    assert any(p == "What is Plasma?" for p in prompts)
    assert any("orange" in (s.target or "").lower() for s in samples)
    assert any("metal bird" in p for p in prompts)
    assert not any("And That" in p for p in prompts)
    assert not any("What are the questions" in p for p in prompts)


def test_dedupe_prefers_definition_over_teaching() -> None:
    from allm.data.base import Sample

    teaching = Sample(
        id="a",
        input="What is Plasma?",
        target="short teaching",
        metadata={"sample_kind": "teaching"},
    )
    definition = Sample(
        id="b",
        input="What is Plasma?",
        target="Plasma is the way the Universe works in full detail here",
        metadata={"sample_kind": "definition"},
    )
    merged = dedupe_samples([teaching, definition])
    assert len(merged) == 1
    assert merged[0].metadata["sample_kind"] == "definition"


def test_parse_sample_kinds_exam_alias() -> None:
    kinds = parse_sample_kinds("exam")
    assert kinds == frozenset({"definition", "we_call", "compact"})


def test_mk_corpus_from_real_files() -> None:
    mk_dir = Path(__file__).resolve().parents[1] / "transcripts" / "Kids" / "cleaned" / "mk"
    if not mk_dir.is_dir():
        return
    samples = mk_corpus_to_samples(mk_dir, kinds=frozenset({"definition", "we_call", "compact"}))
    assert len(samples) >= 40
    assert all(s.target for s in samples)
    assert not any("What is And That" in s.input for s in samples)
    assert not any("explain:" in s.input for s in samples)
