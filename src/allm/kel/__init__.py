"""KEL — Knowledge Evaluation Layer (see KEL.md, project root).

Measures whether the knowledge system is *improving*, not just growing:
six metrics (RCR, CD, GST, CRR, LG, CRE), the composite Graph Health
Score, persisted time series with trends, and detectors for the four
failure modes (false compression, dead knowledge growth, conflict
accumulation, static illusion). Measurement-only: KEL never modifies
knowledge.
"""

from allm.kel.layer import KnowledgeEvaluationLayer
from allm.kel.types import Finding, GraphSnapshot, KELConfig, KELReport

__all__ = [
    "KnowledgeEvaluationLayer",
    "Finding",
    "GraphSnapshot",
    "KELConfig",
    "KELReport",
]
