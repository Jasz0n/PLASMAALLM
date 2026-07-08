"""Persisted Teacher visual brief approvals (M25)."""

from __future__ import annotations

import json

from allm.researcher.multimodal_types import DistilledVisualBrief, TeacherVisualApproval
from allm.storage.base import RecordStore

NAMESPACE = "teacher_visual_approvals"


class VisualApprovalStore:
    """Append-only store for Teacher decisions on distilled visual briefs."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def save(self, approval: TeacherVisualApproval, *, reason: str = "teacher review") -> None:
        """Persist one approval decision."""
        self._store.put(
            NAMESPACE,
            approval.brief_id,
            json.loads(approval.model_dump_json()),
            reason=reason,
        )

    def get(self, brief_id: str) -> TeacherVisualApproval | None:
        """Latest approval for one brief, if any."""
        record = self._store.get(NAMESPACE, brief_id)
        if record is None:
            return None
        return TeacherVisualApproval.model_validate(record.value)

    def all_approvals(self) -> tuple[TeacherVisualApproval, ...]:
        """Latest approval per brief id."""
        rows: list[TeacherVisualApproval] = []
        for key in self._store.keys(NAMESPACE):
            record = self._store.get(NAMESPACE, key)
            if record is None:
                continue
            rows.append(TeacherVisualApproval.model_validate(record.value))
        return tuple(sorted(rows, key=lambda row: row.brief_id))

    def pending_briefs(
        self,
        briefs: tuple[DistilledVisualBrief, ...],
    ) -> tuple[DistilledVisualBrief, ...]:
        """Briefs that have no stored Teacher decision yet."""
        decided = {approval.brief_id for approval in self.all_approvals()}
        return tuple(brief for brief in briefs if brief.brief_id not in decided)
