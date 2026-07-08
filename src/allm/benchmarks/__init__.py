"""Benchmarks: multi-dimensional suite (M43) + system report (M47)."""

from allm.benchmarks.suite import BenchmarkDimension, BenchmarkResult, BenchmarkSuite
from allm.benchmarks.system_report import (
    STANDARD_CORPORA,
    CorpusReport,
    SystemReport,
    run_system_benchmark,
)

__all__ = [
    "BenchmarkDimension",
    "BenchmarkResult",
    "BenchmarkSuite",
    "STANDARD_CORPORA",
    "CorpusReport",
    "SystemReport",
    "run_system_benchmark",
]
