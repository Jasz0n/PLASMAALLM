"""Contribution proposals (Roadmap M49): humans hold the only pen.

The invariant this module *is*: **nothing leaves the system without a
human approval record.** A candidate patch that survived its trial run
becomes a proposed contribution; only an explicit approval — with a
named human reviewer and a reason — unlocks applying it to a working
tree. There is deliberately no push, no remote, no auto-merge anywhere
in this module or its dependencies; applying writes one local file and
records who allowed it, versioned like every other belief.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path

from allm.core.logging import get_logger
from allm.practice.repo_tasks import CandidatePatch
from allm.practice.types import PracticeRun
from allm.storage.base import RecordStore

logger = get_logger("practice.contribution")

NAMESPACE = "contributions"

ContributionStatus = Literal["proposed", "approved", "rejected", "applied"]


class ApprovalError(PermissionError):
    """Raised when an action requires a human approval record that is absent."""


class Contribution(BaseModel):
    """One proposed change with its evidence trail and review state."""

    model_config = ConfigDict(frozen=True)

    id: str
    patch: CandidatePatch
    test_selector: str
    trial_run_id: str
    trial_outcome: str
    status: ContributionStatus = "proposed"
    reviewer: str | None = None
    review_reason: str | None = None
    proposed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_at: datetime | None = None
    applied_at: datetime | None = None


class ContributionBoard:
    """Versioned lifecycle: proposed → approved/rejected → applied."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def propose(
        self, patch: CandidatePatch, *, test_selector: str, trial: PracticeRun
    ) -> Contribution:
        """A trialed patch becomes a reviewable proposal (evidence attached)."""
        contribution = Contribution(
            id=f"contrib_{patch.id.removeprefix('patch_')}",
            patch=patch,
            test_selector=test_selector,
            trial_run_id=trial.id,
            trial_outcome=trial.outcome,
        )
        self._put(contribution, f"proposed by {patch.author}: trial {trial.outcome[:40]!r}")
        return contribution

    def get(self, contribution_id: str) -> Contribution | None:
        record = self._store.get(NAMESPACE, contribution_id)
        return None if record is None else Contribution.model_validate(record.value)

    def all(self) -> list[Contribution]:
        rows = [self.get(key) for key in self._store.keys(NAMESPACE)]
        return sorted((r for r in rows if r), key=lambda c: c.proposed_at)

    def approve(self, contribution_id: str, *, reviewer: str, reason: str) -> Contribution:
        """A named human takes responsibility; the record is the permission."""
        return self._review(contribution_id, "approved", reviewer=reviewer, reason=reason)

    def reject(self, contribution_id: str, *, reviewer: str, reason: str) -> Contribution:
        """Rejection with a reason is a failure sample, not a dead end."""
        return self._review(contribution_id, "rejected", reviewer=reviewer, reason=reason)

    def apply(self, contribution_id: str, repo_dir: Path | str) -> Path:
        """Write the approved patch into the working tree — and only then.

        Raises :class:`ApprovalError` without an approval record. Never
        commits, never pushes: what happens after the file lands is a
        human's git history, under a human's name.
        """
        contribution = self.get(contribution_id)
        if contribution is None:
            raise KeyError(f"unknown contribution {contribution_id!r}")
        if contribution.status != "approved":
            raise ApprovalError(
                f"contribution {contribution_id} is {contribution.status!r} — "
                "nothing leaves the system without a human approval record"
            )
        target = Path(repo_dir) / contribution.patch.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contribution.patch.content, encoding="utf-8")
        applied = contribution.model_copy(
            update={"status": "applied", "applied_at": datetime.now(timezone.utc)}
        )
        self._put(
            applied,
            f"applied to {target} under approval by {contribution.reviewer}",
        )
        logger.info("applied %s to %s (approved by %s)", contribution_id, target, contribution.reviewer)
        return target

    def _review(
        self, contribution_id: str, status: ContributionStatus, *, reviewer: str, reason: str
    ) -> Contribution:
        if not reviewer.strip():
            raise ApprovalError("a review needs a named human reviewer")
        if not reason.strip():
            raise ApprovalError("a review needs a reason — it becomes the learning signal")
        contribution = self.get(contribution_id)
        if contribution is None:
            raise KeyError(f"unknown contribution {contribution_id!r}")
        if contribution.status != "proposed":
            raise ApprovalError(
                f"contribution {contribution_id} already {contribution.status!r}"
            )
        reviewed = contribution.model_copy(
            update={
                "status": status,
                "reviewer": reviewer,
                "review_reason": reason,
                "reviewed_at": datetime.now(timezone.utc),
            }
        )
        self._put(reviewed, f"{status} by {reviewer}: {reason[:60]}")
        return reviewed

    def _put(self, contribution: Contribution, reason: str) -> None:
        self._store.put(
            NAMESPACE,
            contribution.id,
            contribution.model_dump(mode="json"),
            reason=reason,
        )
