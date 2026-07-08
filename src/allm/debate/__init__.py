"""Debate Engine: independent answers, clustered, with measured
disagreement. Confidence-weighted verdicts; unresolved debates convert
into learning tasks via :meth:`DebateResult.to_learning_sample`.
"""

from allm.debate.engine import DebateEngine
from allm.debate.evidence import DebateEvidenceResolution, resolve_debate_evidence
from allm.debate.types import Cluster, DebateResult, Position

__all__ = [
    "DebateEngine",
    "Cluster",
    "DebateResult",
    "Position",
    "DebateEvidenceResolution",
    "resolve_debate_evidence",
]
