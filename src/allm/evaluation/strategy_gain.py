"""Marginal learning gain per strategy phase — derived from iteration history."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from allm.loop.history import LearningIterationRecord
from allm.loop.strategy import LearningStrategyName


class StrategyPhaseGain(BaseModel):
    """Held-out and KEL deltas accumulated during one strategy phase."""

    model_config = ConfigDict(frozen=True)

    strategy: LearningStrategyName
    iteration_start: int
    iteration_end: int
    iterations: int
    samples_studied: int
    score_before: float
    score_after: float
    heldout_gain: float
    kel_lg_before: float | None = None
    kel_lg_after: float | None = None
    kel_lg_delta: float | None = None


def compute_marginal_strategy_gains(
    records: list[LearningIterationRecord],
) -> list[StrategyPhaseGain]:
    """Group consecutive iterations by strategy and sum per-phase learning gain."""
    if not records:
        return []

    ordered = sorted(records, key=lambda row: row.iteration)
    phases: list[StrategyPhaseGain] = []
    block: list[LearningIterationRecord] = [ordered[0]]

    for record in ordered[1:]:
        if record.strategy == block[-1].strategy:
            block.append(record)
            continue
        phases.append(_phase_from_block(block))
        block = [record]

    phases.append(_phase_from_block(block))
    return phases


def _phase_from_block(block: list[LearningIterationRecord]) -> StrategyPhaseGain:
    """Collapse one contiguous strategy block into a phase summary."""
    first, last = block[0], block[-1]
    kel_before, kel_after = first.kel_lg, last.kel_lg
    kel_delta = None
    if kel_before is not None and kel_after is not None:
        kel_delta = kel_after - kel_before

    return StrategyPhaseGain(
        strategy=first.strategy,
        iteration_start=first.iteration,
        iteration_end=last.iteration,
        iterations=len(block),
        samples_studied=sum(row.samples_studied for row in block),
        score_before=first.score_before,
        score_after=last.score_after,
        heldout_gain=last.score_after - first.score_before,
        kel_lg_before=kel_before,
        kel_lg_after=kel_after,
        kel_lg_delta=kel_delta,
    )


def format_strategy_gain_report(phases: list[StrategyPhaseGain]) -> str:
    """Human-readable table for experiment logs."""
    if not phases:
        return "  (no strategy phases recorded)"

    lines = [
        "  Strategy          Iters   Samples   Held-out Δ   KEL LG Δ",
        "  " + "-" * 58,
    ]
    for phase in phases:
        iter_span = (
            str(phase.iteration_start)
            if phase.iteration_start == phase.iteration_end
            else f"{phase.iteration_start}-{phase.iteration_end}"
        )
        kel = f"{phase.kel_lg_delta:+.3f}" if phase.kel_lg_delta is not None else "n/a"
        lines.append(
            f"  {phase.strategy:<16}  {iter_span:>5}   {phase.samples_studied:>7}   "
            f"{phase.heldout_gain:+.2f}        {kel}"
        )
    return "\n".join(lines)


def export_strategy_phase_gains(
    path: Path | str,
    records: list[LearningIterationRecord],
) -> Path:
    """Write per-phase marginal gains next to a learning history file."""
    target = Path(path)
    phases = compute_marginal_strategy_gains(records)
    target.write_text(
        json.dumps([phase.model_dump() for phase in phases], indent=2),
        encoding="utf-8",
    )
    return target
