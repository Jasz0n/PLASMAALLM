"""Variable sweeps: the same procedure, different variables (PRACTICE.md).

The module's founding observation, made executable: certain things done
the same way but with different variables return a different outcome —
and that difference is knowledge no text corpus carries. A sweep varies
exactly one variable, holds the rest at their defaults, and earns the
graph a dependency relation either way.
"""

from __future__ import annotations

from typing import Any, Iterable

from allm.core.logging import get_logger
from allm.knowledge.graph import KnowledgeGraph
from allm.knowledge.types import Concept, Evidence
from allm.practice.evidence import run_claim
from allm.practice.executor import SandboxExecutor
from allm.practice.types import PracticeProcedure, PracticeRun, SweepResult

logger = get_logger("practice.sweep")


def next_variable(
    procedure: PracticeProcedure, history: Iterable[PracticeRun]
) -> str:
    """Which variable to vary next: the least-explored one.

    Curiosity as in Plan.md — the most informative experiment is where
    we have looked least. Explored = distinct values observed in
    ``history`` for runs of this procedure; ties break by declaration
    order.
    """
    if not procedure.variables:
        raise ValueError(f"procedure {procedure.id!r} has no variables to explore")
    seen: dict[str, set[str]] = {v.name: set() for v in procedure.variables}
    for run in history:
        if run.procedure_id != procedure.id:
            continue
        for name, value in run.variables.items():
            if name in seen:
                seen[name].add(repr(value))
    return min(procedure.variables, key=lambda v: len(seen[v.name])).name


def run_sweep(
    procedure: PracticeProcedure,
    variable: str,
    *,
    executor: SandboxExecutor | None = None,
    values: tuple[Any, ...] | None = None,
) -> SweepResult:
    """Vary one variable over its candidates, defaults for the rest."""
    spec = procedure.variable(variable)
    sweep_values = values if values is not None else (spec.default, *spec.candidates)
    if len(sweep_values) < 2:
        raise ValueError(
            f"sweeping {variable!r} needs at least two values; declare candidates"
        )
    runner = executor or SandboxExecutor()
    fixed = {k: v for k, v in procedure.defaults().items() if k != variable}
    runs = tuple(runner.run(procedure, {**fixed, variable: value}) for value in sweep_values)
    depends = len({run.outcome for run in runs}) > 1
    logger.info(
        "sweep %s over %s: %d runs, outcome %s on it",
        procedure.id, variable, len(runs), "DEPENDS" if depends else "does not depend",
    )
    return SweepResult(
        procedure_id=procedure.id,
        variable=variable,
        fixed=fixed,
        runs=runs,
        depends=depends,
    )


def record_sweep(
    graph: KnowledgeGraph, procedure: PracticeProcedure, sweep: SweepResult
) -> Concept:
    """Write the sweep's verdict into the knowledge graph, append-only.

    The procedure's concept gains the dependency relation in ``related``
    and one :class:`Evidence` entry per run — every conclusion stays
    traceable to the executions behind it.
    """
    evidence = tuple(
        Evidence(source=run.id, detail=run_claim(run), supports=run.status == "ok")
        for run in sweep.runs
    )
    reason = f"practice sweep over {sweep.variable} ({len(sweep.runs)} runs)"
    if graph.get(procedure.concept_name) is None:
        return graph.add(
            Concept(
                name=procedure.concept_name,
                description=procedure.description,
                related=(sweep.relation,),
                evidence=evidence,
                source="practice-engine",
                usefulness=0.7,
            ),
            reason=reason,
        )
    return graph.revise(
        procedure.concept_name,
        reason=reason,
        add_related=(sweep.relation,),
        add_evidence=evidence,
    )
