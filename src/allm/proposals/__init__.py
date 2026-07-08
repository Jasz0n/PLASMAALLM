"""Experiment Proposals: the human verification loop from smallVision.md.

Debate disagreements, KDP conflicts and planner gaps become proposals;
humans claim them and resolve them with evidence packages; the evidence
— never a decree — decides the outcome and updates graph confidence.
"""

from allm.proposals.board import ProposalBoard, ProposalError
from allm.proposals.types import ExperimentProposal, ProposalStatus, Resolution

__all__ = [
    "ProposalBoard",
    "ProposalError",
    "ExperimentProposal",
    "ProposalStatus",
    "Resolution",
]
