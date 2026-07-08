"""Data Collector: deduplicating, quality-scoring sample pool.

Any source (dataset loaders, failure logs, debate outcomes) feeds the
same pool; web/book/paper acquisition is a future source behind the
same contract.
"""

from allm.collector.pool import Collector, SamplePool, quality

__all__ = ["Collector", "SamplePool", "quality"]
