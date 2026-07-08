"""Append-only recommendation queue for the Teacher."""

from __future__ import annotations

import json

from allm.researcher.types import ResearchRecommendation
from allm.storage.base import RecordStore

NAMESPACE = "researcher_recommendations"
PACKAGES_NAMESPACE = "researcher_packages"


class RecommendationQueue:
    """Persist Researcher recommendations — Teacher reads, never overwritten."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def enqueue(self, recommendation: ResearchRecommendation, *, reason: str = "discovered") -> None:
        """Append one recommendation."""
        self._store.put(
            NAMESPACE,
            f"{recommendation.topic}::{recommendation.package_id}",
            json.loads(recommendation.model_dump_json()),
            reason=reason,
        )

    def packages(self) -> list:
        """All stored Knowledge Packages."""
        from allm.researcher.types import KnowledgePackage

        rows = []
        for key in self._store.keys(PACKAGES_NAMESPACE):
            record = self._store.get(PACKAGES_NAMESPACE, key)
            if record is None:
                continue
            rows.append(KnowledgePackage.model_validate(record.value))
        return rows

    def active(self, *, limit: int | None = None) -> list[ResearchRecommendation]:
        """Latest recommendation per topic, highest priority first."""
        by_topic: dict[str, ResearchRecommendation] = {}
        for key in self._store.keys(NAMESPACE):
            record = self._store.get(NAMESPACE, key)
            if record is None:
                continue
            rec = ResearchRecommendation.model_validate(record.value)
            existing = by_topic.get(rec.topic)
            if existing is None or rec.priority > existing.priority:
                by_topic[rec.topic] = rec
        ordered = sorted(by_topic.values(), key=lambda row: (-row.priority, row.topic))
        return ordered if limit is None else ordered[:limit]

    def store_package(self, package, *, reason: str = "packaged") -> None:
        """Persist a Knowledge Package snapshot."""
        self._store.put(
            PACKAGES_NAMESPACE,
            package.id,
            json.loads(package.model_dump_json()),
            reason=reason,
        )
