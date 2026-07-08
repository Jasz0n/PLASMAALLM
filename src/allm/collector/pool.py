"""Sample pool: deduplicating, quality-scoring collection.

Phase 10 scope: the pool ingests samples from any source (dataset
loaders, failure logs, debate outcomes), removes duplicates by
normalised input, scores quality, and serves the best samples per
topic. Web/book/paper acquisition is a future source that feeds the
same pool; the dedupe/quality/serve contract is the stable part.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence, runtime_checkable

from allm.core.logging import get_logger
from allm.data.base import Sample
from allm.exam.grading import normalise

logger = get_logger("collector.pool")


@runtime_checkable
class Collector(Protocol):
    """Serves training samples for requested topics."""

    def collect(
        self, topics: Sequence[str] | None = None, limit: int | None = None
    ) -> list[Sample]: ...


def quality(sample: Sample) -> float:
    """Naive quality score, deliberately transparent.

    Labelled samples are directly trainable (1.0); unlabelled ones are
    only research prompts (0.4). Source-based scoring (provenance,
    duplication across sources) extends this function, not the pool.
    """
    return 1.0 if sample.target is not None else 0.4


class SamplePool:
    """In-memory deduplicating pool of training samples."""

    def __init__(self) -> None:
        self._samples: dict[str, Sample] = {}

    def ingest(self, samples: Iterable[Sample]) -> int:
        """Add samples, deduplicating by normalised input.

        A labelled sample replaces an unlabelled duplicate (it knows
        strictly more); otherwise first-in wins. Returns how many
        entries were added or upgraded.
        """
        changed = 0
        for sample in samples:
            key = normalise(sample.input)
            existing = self._samples.get(key)
            if existing is None:
                self._samples[key] = sample
                changed += 1
            elif existing.target is None and sample.target is not None:
                self._samples[key] = sample
                changed += 1
        logger.debug("ingested %d new/upgraded samples (pool=%d)", changed, len(self._samples))
        return changed

    def collect(
        self,
        topics: Sequence[str] | None = None,
        limit: int | None = None,
        *,
        kinds: Sequence[str] | None = None,
    ) -> list[Sample]:
        """Best samples first, optionally filtered by topic and sample kind."""
        wanted = None if topics is None else set(topics)
        kind_set = None if kinds is None else set(kinds)
        matches = [
            s
            for s in self._samples.values()
            if wanted is None or s.metadata.get("topic", "general") in wanted
        ]
        if kind_set is not None:
            matches = [
                s
                for s in matches
                if (s.metadata or {}).get("sample_kind", "teaching") in kind_set
            ]
        matches.sort(key=lambda s: (-quality(s), s.id))
        return matches if limit is None else matches[:limit]

    def topics(self) -> list[str]:
        """Distinct topics present in the pool."""
        return sorted({s.metadata.get("topic", "general") for s in self._samples.values()})

    def __len__(self) -> int:
        return len(self._samples)
