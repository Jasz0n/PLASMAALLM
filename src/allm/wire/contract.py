"""Assemble the versioned wire contract from the real models.

``WIRE_VERSION`` is the contract's own semantic version, deliberately
separate from ``allm.__version__``: the engine can be refactored, and
new *optional* fields can appear, without moving it; only a
backward-incompatible change to what a client must send or can rely on
bumps the major. The schemas themselves are generated from the pydantic
models, so "published" and "implemented" cannot fall out of step — a CI
drift guard (``test_published_wire_contract_is_current``) proves it.
"""

from __future__ import annotations

from typing import get_args

import allm
from allm.api.schemas import (
    ClaimRequest,
    ConceptSummary,
    DocumentSubmission,
    EvidenceSubmission,
    ResolveRequest,
)
from allm.events import Event
from allm.evidence.types import ConfidenceBreakdown, Outcome, PackageKind

# Bump the major only on a backward-incompatible change to the contract
# (a required field added/removed, a type narrowed, a vocabulary value
# retired). Additive, optional changes bump the minor.
WIRE_VERSION = "1.0.0"

# What a contributor sends in.
_REQUESTS = {
    "EvidenceSubmission": EvidenceSubmission,
    "DocumentSubmission": DocumentSubmission,
    "ClaimRequest": ClaimRequest,
    "ResolveRequest": ResolveRequest,
}

# What the core sends back and streams out.
_RESPONSES = {
    "ConfidenceBreakdown": ConfidenceBreakdown,
    "ConceptSummary": ConceptSummary,
    "Event": Event,
}


def wire_contract() -> dict:
    """The whole contract as one JSON-serialisable document."""
    return {
        "wire_version": WIRE_VERSION,
        "generated_by": allm.__version__,
        "title": "ALLM evidence wire format",
        "description": (
            "The public contract platform teams build against: how to submit "
            "evidence and documents, drive the proposal lifecycle, and read "
            "confidence and the event feed. Versioned independently of the "
            "engine; schemas are generated from the implementation."
        ),
        "vocabularies": {
            "PackageKind": list(get_args(PackageKind)),
            "Outcome": list(get_args(Outcome)),
        },
        "requests": {name: model.model_json_schema() for name, model in _REQUESTS.items()},
        "responses": {name: model.model_json_schema() for name, model in _RESPONSES.items()},
    }
