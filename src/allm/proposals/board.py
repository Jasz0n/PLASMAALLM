"""The proposal board: where AI questions meet human evidence.

Design decisions
----------------
- Proposals are versioned records (namespace ``proposals``); every
  transition (open -> claimed -> resolved) is a new version with a
  reason, so the full negotiation history survives.
- Duplicate protection: proposing the same (concept, question) while
  an equivalent proposal is unresolved returns the existing one —
  humans should never see the same ask twice.
- ``resolve`` accepts evidence packages, pushes them through the
  :class:`EvidenceBinder` (graph confidence updates exactly like any
  other submission), and derives the proposal outcome from those
  packages' evidential confidence. Nobody resolves a proposal by
  decree; the evidence decides.
- Factory helpers turn debate results, KDP conflicts and roadmap items
  into proposals, closing smallVision.md's loop mechanically.
"""

from __future__ import annotations

from allm.core.logging import get_logger
from allm.debate.types import DebateResult
from allm.evidence.confidence import evidential_confidence
from allm.evidence.ledger import EvidenceBinder
from allm.evidence.types import EvidencePackage
from allm.kdp.types import ConflictNode, content_hash
from allm.planner.types import RoadmapItem
from allm.proposals.types import ExperimentProposal, ProposalOrigin, Resolution
from allm.storage.base import RecordStore

logger = get_logger("proposals.board")

NAMESPACE = "proposals"

SUPPORT_THRESHOLD = 0.55
CHALLENGE_THRESHOLD = 0.45


class ProposalError(ValueError):
    """Raised on invalid proposal transitions."""


class ProposalBoard:
    """Versioned proposal lifecycle over the record store."""

    def __init__(self, store: RecordStore, binder: EvidenceBinder) -> None:
        self._store = store
        self._binder = binder

    # -- creating ---------------------------------------------------------

    def propose(
        self,
        concept: str,
        question: str,
        *,
        rationale: str,
        origin: ProposalOrigin = "manual",
    ) -> ExperimentProposal:
        proposal_id = f"prop_{content_hash(concept, question)}"
        existing = self.get(proposal_id)
        if existing is not None and existing.status != "resolved":
            return existing  # same ask, still pending: no duplicates
        proposal = ExperimentProposal(
            id=proposal_id,
            concept=concept,
            question=question,
            rationale=rationale,
            origin=origin,
        )
        self._put(proposal, f"proposed ({origin}): {question}")
        return proposal

    def from_debate(self, result: DebateResult) -> ExperimentProposal:
        return self.propose(
            result.question.topic,
            result.question.prompt,
            rationale=(
                f"students disagreed (disagreement {result.disagreement:.2f}; "
                f"leading answer {result.verdict!r})"
            ),
            origin="debate",
        )

    def from_conflict(self, conflict: ConflictNode) -> ExperimentProposal:
        return self.propose(
            conflict.concept,
            f"Which interpretation of {conflict.concept} holds up experimentally?",
            rationale=(
                f"sources disagree: {conflict.interpretation_a[:80]!r} "
                f"vs {conflict.interpretation_b[:80]!r}"
            ),
            origin="conflict",
        )

    def from_roadmap_item(self, item: RoadmapItem) -> ExperimentProposal:
        return self.propose(
            item.topic,
            f"Gather ground-truth evidence about {item.topic}",
            rationale=f"planner gap: {item.reason}",
            origin="planner",
        )

    # -- lifecycle -----------------------------------------------------------

    def claim(self, proposal_id: str, contributor: str) -> ExperimentProposal:
        proposal = self._require(proposal_id)
        if proposal.status != "open":
            raise ProposalError(f"{proposal_id} is {proposal.status}, not open")
        claimed = proposal.model_copy(
            update={"status": "claimed", "claimed_by": contributor}
        )
        self._put(claimed, f"claimed by {contributor}")
        return claimed

    def resolve(
        self, proposal_id: str, packages: list[EvidencePackage]
    ) -> ExperimentProposal:
        """Submit the packages and let their evidence settle the proposal."""
        proposal = self._require(proposal_id)
        if proposal.status == "resolved":
            raise ProposalError(f"{proposal_id} is already resolved")
        relevant = [p for p in packages if p.concept == proposal.concept]
        if not relevant:
            raise ProposalError(
                f"no submitted package targets concept {proposal.concept!r}"
            )
        for package in packages:
            self._binder.submit(package)

        breakdown = evidential_confidence(proposal.concept, relevant)
        if breakdown.value >= SUPPORT_THRESHOLD:
            outcome = "supported"
        elif breakdown.value <= CHALLENGE_THRESHOLD:
            outcome = "challenged"
        else:
            outcome = "inconclusive"
        resolved = proposal.model_copy(
            update={
                "status": "resolved",
                "resolution": Resolution(
                    outcome=outcome,
                    package_ids=tuple(sorted(p.id for p in relevant)),
                ),
            }
        )
        self._put(
            resolved,
            f"resolved {outcome} by {len(relevant)} package(s) "
            f"(evidence confidence {breakdown.value:.2f})",
        )
        logger.info("proposal %s resolved: %s", proposal_id, outcome)
        return resolved

    # -- reading -----------------------------------------------------------------

    def get(self, proposal_id: str) -> ExperimentProposal | None:
        record = self._store.get(NAMESPACE, proposal_id)
        return None if record is None else ExperimentProposal.model_validate(record.value)

    def proposals(self, status: str | None = None) -> list[ExperimentProposal]:
        """All proposals (optionally by status), oldest first."""
        result = [
            p
            for key in self._store.keys(NAMESPACE)
            if (p := self.get(key)) is not None and (status is None or p.status == status)
        ]
        result.sort(key=lambda p: (p.created_at, p.id))
        return result

    # -- internals ------------------------------------------------------------------

    def _require(self, proposal_id: str) -> ExperimentProposal:
        proposal = self.get(proposal_id)
        if proposal is None:
            raise ProposalError(f"unknown proposal {proposal_id!r}")
        return proposal

    def _put(self, proposal: ExperimentProposal, reason: str) -> None:
        self._store.put(
            NAMESPACE, proposal.id, proposal.model_dump(mode="json"), reason=reason
        )
