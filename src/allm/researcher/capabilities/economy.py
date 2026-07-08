"""L6 — Knowledge economy / provider reputation ledger."""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.capabilities.base import (
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.storage.base import RecordStore

LEDGER_NAMESPACE = "researcher_providers"


class ProviderLedgerEntry(BaseModel):
    """Append-only provider performance record."""

    model_config = ConfigDict(frozen=True)

    provider_id: str
    packages_submitted: int = 0
    recommendations_enqueued: int = 0
    conflicts_found: int = 0
    reputation_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ProviderReputationLedger:
    """Append-only ledger for provider trust evolution."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def record_cycle(
        self,
        provider_id: str,
        *,
        packages: int,
        recommendations: int,
        conflicts: int,
        reputation_score: float,
    ) -> ProviderLedgerEntry:
        previous = self.latest(provider_id)
        entry = ProviderLedgerEntry(
            provider_id=provider_id,
            packages_submitted=(previous.packages_submitted if previous else 0) + packages,
            recommendations_enqueued=(previous.recommendations_enqueued if previous else 0)
            + recommendations,
            conflicts_found=(previous.conflicts_found if previous else 0) + conflicts,
            reputation_score=reputation_score,
        )
        self._store.put(
            LEDGER_NAMESPACE,
            provider_id,
            json.loads(entry.model_dump_json()),
            reason="provider cycle",
        )
        return entry

    def latest(self, provider_id: str) -> ProviderLedgerEntry | None:
        record = self._store.get(LEDGER_NAMESPACE, provider_id)
        if record is None:
            return None
        return ProviderLedgerEntry.model_validate(record.value)

    def leaderboard(self) -> list[ProviderLedgerEntry]:
        rows = []
        for key in self._store.keys(LEDGER_NAMESPACE):
            record = self._store.get(LEDGER_NAMESPACE, key)
            if record is not None:
                rows.append(ProviderLedgerEntry.model_validate(record.value))
        return sorted(rows, key=lambda row: -row.reputation_score)


class EconomyLedgerCapability:
    """L6 — record provider submissions and acceptance."""

    level = 6
    name = "economy.ledger"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        ledger = ProviderReputationLedger(ctx.store)
        entries: list[ProviderLedgerEntry] = []

        for discovery in pipeline.discoveries:
            pkg_count = sum(1 for p in pipeline.packages if p.provider == discovery.provider_id)
            rec_count = sum(
                1 for r in pipeline.recommendations if r.provider == discovery.provider_id
            )
            conflicts = sum(
                len(p.conflicts)
                for p in pipeline.packages
                if p.provider == discovery.provider_id
            )
            entry = ledger.record_cycle(
                discovery.provider_id,
                packages=pkg_count,
                recommendations=rec_count,
                conflicts=conflicts,
                reputation_score=discovery.reputation_score,
            )
            entries.append(entry)

        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(entries),
                duration_ms=round(elapsed, 2),
            ),
            artifacts={"ledger": entries},
        )
