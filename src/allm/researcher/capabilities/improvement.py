"""L7 — Self-improvement via capability yield tracking."""

from __future__ import annotations

import time

from allm.researcher.capabilities.base import (
    METRICS_NAMESPACE,
    CapabilityContext,
    CapabilityMetrics,
    CapabilityResult,
    PipelineState,
)
from allm.researcher.capabilities.planning import StrategyHints


class ImprovementCapability:
    """L7 — derive strategy hints from capability performance history."""

    level = 7
    name = "improvement.reflect"

    def run(self, ctx: CapabilityContext, pipeline: PipelineState) -> CapabilityResult:
        started = time.perf_counter()
        yields: dict[str, list[int]] = {}
        for key in ctx.store.keys(METRICS_NAMESPACE):
            record = ctx.store.get(METRICS_NAMESPACE, key)
            if record is None:
                continue
            name = record.value.get("capability", "")
            if not name.startswith("discovery."):
                continue
            yields.setdefault(name, []).append(int(record.value.get("yield_count", 0)))

        skip: list[str] = []
        prefer: list[str] = []
        notes: list[str] = []

        provider_map = {
            "discovery.workshop": "kids-workshops",
            "discovery.software": "software-fixture",
        }
        for cap_name, counts in yields.items():
            if len(counts) < 2:
                continue
            recent = counts[-3:]
            if sum(recent) == 0:
                provider = provider_map.get(cap_name)
                if provider:
                    skip.append(provider)
                    notes.append(f"skip {provider}: zero yield in last {len(recent)} runs")
            elif sum(recent) >= 3:
                provider = provider_map.get(cap_name)
                if provider:
                    prefer.append(provider)

        hints = StrategyHints(
            skip_providers=tuple(sorted(set(skip))),
            prefer_providers=tuple(sorted(set(prefer))),
            notes=tuple(notes),
        )
        elapsed = (time.perf_counter() - started) * 1000
        return CapabilityResult(
            capability=self.name,
            metrics=CapabilityMetrics(
                capability=self.name,
                level=self.level,
                yield_count=len(notes),
                duration_ms=round(elapsed, 2),
                notes="; ".join(notes) if notes else "no strategy changes",
            ),
            artifacts={"strategy_hints": hints},
        )

    @staticmethod
    def load_strategy_hints(store) -> "StrategyHints | None":
        """Load strategy hints from the dedicated namespace or metric history."""
        from allm.researcher.capabilities.planning import StrategyHints

        record = store.get("researcher_strategy_hints", "latest")
        if record is not None:
            notes = tuple(record.value.get("notes", ()))
            return StrategyHints(notes=notes)

        skip: set[str] = set()
        notes: list[str] = []
        provider_map = {
            "discovery.workshop": "kids-workshops",
            "discovery.software": "software-fixture",
        }
        yields: dict[str, list[int]] = {}
        for key in store.keys(METRICS_NAMESPACE):
            record = store.get(METRICS_NAMESPACE, key)
            if record is None:
                continue
            name = record.value.get("capability", "")
            if name in provider_map:
                yields.setdefault(name, []).append(int(record.value.get("yield_count", 0)))

        for cap_name, counts in yields.items():
            if len(counts) >= 2 and sum(counts[-3:]) == 0:
                provider = provider_map[cap_name]
                skip.add(provider)
                notes.append(f"skip {provider}: zero yield")

        if not notes and not skip:
            return None
        return StrategyHints(skip_providers=tuple(sorted(skip)), notes=tuple(notes))
