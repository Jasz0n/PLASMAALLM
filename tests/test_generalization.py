"""Tests for held-out generalization diagnosis."""

from allm.data.base import Sample
from allm.evaluation.generalization import diagnose_holdout_gap, prompt_overlap


def _sample(sample_id: str, prompt: str, answer: str, source: str, kind: str = "definition") -> Sample:
    return Sample(
        id=sample_id,
        input=prompt,
        target=answer,
        metadata={"source": source, "sample_kind": kind, "topic": "kids-plasma"},
    )


def test_exact_prompt_detected() -> None:
    train = [_sample("t1", "What is Plasma?", "Plasma is matter.", "knowledgeSeekerWorkshop1.txt")]
    holdout = [_sample("h1", "What is Plasma?", "Plasma is energy.", "knowledgeSeekerWorkshop13.txt")]
    report = diagnose_holdout_gap(train, holdout)
    assert report.exact_prompt_matches == 1
    assert report.samples[0].category == "exact_prompt"


def test_novel_lexical_holdout() -> None:
    train = [_sample("t1", "What is Plasma?", "Plasma is matter.", "knowledgeSeekerWorkshop1.txt")]
    holdout = [
        _sample(
            "h1",
            "How do magnetic fields interact with graphene layers?",
            "Opposite pull.",
            "knowledgeSeekerWorkshop14.txt",
        )
    ]
    report = diagnose_holdout_gap(train, holdout)
    assert report.novel_lexical == 1
    assert report.samples[0].train_overlap < 0.2


def test_prompt_overlap_symmetric() -> None:
    assert prompt_overlap("What is Plasma?", "what is plasma") == 1.0
