"""Compression Engine: evidence-preserving abstraction.

Groups of concepts resting on identical foundations become higher-level
principle concepts (members kept and linked, evidence unioned). An
optional performance probe retracts principles that hurt predictive
performance — retraction is a status change with a reason, never a
deletion.
"""

from allm.compression.engine import (
    CompressionEngine,
    CompressionOutcome,
    MergeProposal,
    PerformanceProbe,
)

__all__ = ["CompressionEngine", "CompressionOutcome", "MergeProposal", "PerformanceProbe"]
