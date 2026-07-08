"""Tests for workshop hold-out splits."""

from allm.data.base import Sample
from allm.kdp.holdout import split_samples_holdout, workshop_number


def test_workshop_number_parses_typo_and_standard_names() -> None:
    assert workshop_number("knowleedSeekerWorkshop1.txt") == 1
    assert workshop_number("knowledgeSeekerWorkshop13.txt") == 13


def test_split_holdout_by_workshop() -> None:
    samples = [
        Sample(id="a", input="q1", target="a1", metadata={"source": "knowleedSeekerWorkshop1.txt"}),
        Sample(id="b", input="q2", target="a2", metadata={"source": "knowledgeSeekerWorkshop12.txt"}),
        Sample(id="c", input="q3", target="a3", metadata={"source": "knowledgeSeekerWorkshop13.txt"}),
        Sample(id="d", input="q4", target="a4", metadata={"source": "knowledgeSeekerWorkshop22.txt"}),
    ]
    train, holdout = split_samples_holdout(samples, holdout_after=13)
    assert len(train) == 2
    assert len(holdout) == 2
    assert {s.id for s in holdout} == {"c", "d"}
