"""Tests for KDP concept naming quality heuristics."""

from allm.kdp.concept_quality import concept_quality_report, is_noisy_concept


def test_noisy_fragment_detected() -> None:
    assert is_noisy_concept("And That")
    assert is_noisy_concept("And If You Rotate It As A Plasma")
    assert not is_noisy_concept("Plasma")
    assert not is_noisy_concept("Magnetic Field")


def test_quality_report_counts() -> None:
    report = concept_quality_report(["Plasma", "And That", "Magnetic Field"])
    assert report["total"] == 3
    assert report["noisy"] == 1
    assert report["clean"] == 2
