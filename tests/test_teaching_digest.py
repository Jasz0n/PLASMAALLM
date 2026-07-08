"""Tests for LLM teaching digest parsing (no model calls)."""

from allm.kdp.teaching_digest import (
    build_digest_prompt,
    chunk_turns,
    extract_speaker_turns,
    parse_digest_response,
    render_digest,
)

SAMPLE_CLEANED = """RC: Welcome to the workshop.

MK: Magnets and magnetic fields are the backbone of creation.
Nothing can exist without magnetic fields.

KD: Thank you Mr Keshe.

MK: A Plasma should be an orange in rotation.
"""

SAMPLE_RESPONSE = """TITLE: Plasma as rotating orange
SUMMARY: MK explains magnetic fields and introduces Plasma using an orange analogy.
KEY_CONCEPTS:
- Magnetic fields
- Plasma rotation
FACTS_FOR_KIDS:
- Nothing exists without magnetic fields
MISCONCEPTIONS_ADDRESSED:
- Magnets are not flat fridge shapes in the Universe
DEMOS_AND_ANALOGIES:
- Orange rotating like a Plasma
FOLLOW_UP_QUESTIONS:
- Why is a Plasma like an orange?
"""


def test_extract_mk_turns() -> None:
    turns = extract_speaker_turns(SAMPLE_CLEANED, speaker="MK")
    assert len(turns) == 2
    assert "orange" in turns[1].lower()


def test_chunk_turns_splits_long_sessions() -> None:
    turns = ["a" * 3000, "b" * 3000, "c" * 1000]
    chunks = chunk_turns(turns, max_chars=5500)
    assert len(chunks) == 2


def test_build_prompt_includes_mk_only() -> None:
    prompt = build_digest_prompt("workshop3.txt", extract_speaker_turns(SAMPLE_CLEANED))
    assert "magnetic fields" in prompt.lower()
    assert "Welcome to the workshop" not in prompt


def test_parse_digest_response() -> None:
    digest = parse_digest_response("workshop3.txt", SAMPLE_RESPONSE)
    assert digest.title == "Plasma as rotating orange"
    assert len(digest.key_concepts) == 2
    assert len(digest.follow_up_questions) == 1


def test_parse_digest_markdown_bold_headers() -> None:
    raw = """**TITLE:** Plasma fields
**SUMMARY:** MK taught plasma basics.
**KEY_CONCEPTS:**
- Magnetic fields
**FOLLOW_UP_QUESTIONS:**
- What is plasma?
"""
    digest = parse_digest_response("w3.txt", raw)
    assert digest.title == "Plasma fields"
    assert len(digest.key_concepts) == 1
