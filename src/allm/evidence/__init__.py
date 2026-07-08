"""Evidence Packages: contributions as structured, traceable evidence.

The core of smallVision.md: every human contribution is an
:class:`EvidencePackage` (claim, artifacts, measurements, conditions,
reproduction steps, outcome), stored append-only, linked to graph
concepts, with confidence computed from reproducible results — never
from popularity or authority.
"""

from allm.evidence.confidence import evidential_confidence, package_weight
from allm.evidence.ledger import EvidenceBinder, EvidenceLedger
from allm.evidence.types import (
    Artifact,
    ConfidenceBreakdown,
    EvidencePackage,
    Outcome,
    PackageKind,
)

__all__ = [
    "Artifact",
    "ConfidenceBreakdown",
    "EvidenceBinder",
    "EvidenceLedger",
    "EvidencePackage",
    "Outcome",
    "PackageKind",
    "evidential_confidence",
    "package_weight",
]
