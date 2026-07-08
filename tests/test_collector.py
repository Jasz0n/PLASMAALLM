"""Tests for allm.collector."""

from allm.collector import Collector, SamplePool, quality
from allm.data.base import Sample


def sample(sid: str, text: str, target: str | None = "x", topic: str = "math") -> Sample:
    return Sample(id=sid, input=text, target=target, metadata={"topic": topic})


def test_satisfies_protocol() -> None:
    assert isinstance(SamplePool(), Collector)


def test_ingest_deduplicates_by_normalised_input() -> None:
    pool = SamplePool()
    added = pool.ingest([sample("a", "2+2?"), sample("b", "  2+2? ")])
    assert added == 1
    assert len(pool) == 1


def test_labelled_upgrades_unlabelled_duplicate() -> None:
    pool = SamplePool()
    pool.ingest([sample("a", "2+2?", target=None)])
    assert pool.ingest([sample("b", "2+2?", target="4")]) == 1
    assert pool.collect()[0].target == "4"
    # but a second labelled duplicate does not replace the first
    assert pool.ingest([sample("c", "2+2?", target="four")]) == 0


def test_collect_filters_by_topic_and_ranks_by_quality() -> None:
    pool = SamplePool()
    pool.ingest(
        [
            sample("a", "2+2?", topic="math"),
            sample("b", "why is the sky blue?", target=None, topic="physics"),
            sample("c", "3*3?", topic="math"),
        ]
    )
    assert [s.id for s in pool.collect(topics=["math"])] == ["a", "c"]
    everything = pool.collect()
    assert everything[-1].id == "b"  # unlabelled ranks last
    assert quality(everything[0]) > quality(everything[-1])


def test_collect_limit_and_topics() -> None:
    pool = SamplePool()
    pool.ingest([sample(str(i), f"q{i}?") for i in range(5)])
    assert len(pool.collect(limit=2)) == 2
    assert pool.topics() == ["math"]
