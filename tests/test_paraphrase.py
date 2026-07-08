"""Tests for definition prompt paraphrasing."""

from allm.exam.paraphrase import paraphrase_definition_prompt


def test_paraphrase_what_is_changes_wording() -> None:
    original = "What is Plasma?"
    variant = paraphrase_definition_prompt(original, variant=0)
    assert variant != original
    assert "Plasma" in variant


def test_paraphrase_we_call() -> None:
    original = "What do the Kids workshops call metal bird?"
    variant = paraphrase_definition_prompt(original, variant=0)
    assert "metal bird" in variant
    assert variant != original
