"""Runs become evidence packages (PRACTICE.md Stage 4).

Every execution is a first-class contribution: claim, exact program,
variables, observed outcome — inspectable with the same machinery as
human evidence. Reproduction failures become preserved conflicts.
"""

from __future__ import annotations

import sys

from allm.evidence.types import EvidencePackage
from allm.kdp.types import ConflictNode
from allm.practice.types import PracticeProcedure, PracticeRun

PRACTICE_CONTRIBUTOR = "practice-engine"


def run_claim(run: PracticeRun) -> str:
    variables = ", ".join(f"{k}={v!r}" for k, v in sorted(run.variables.items()))
    return f"procedure {run.procedure_id} with {variables} yields {run.outcome!r}"


def run_to_package(procedure: PracticeProcedure, run: PracticeRun) -> EvidencePackage:
    """One run → one evidence package, content-addressed like the run."""
    return EvidencePackage.build(
        claim=run_claim(run),
        concept=procedure.concept_name,
        contributor=PRACTICE_CONTRIBUTOR,
        kind="experiment",
        outcome="supported" if run.status == "ok" else "challenged",
        measurements={
            "variables": dict(sorted(run.variables.items())),
            "status": run.status,
            "outcome": run.outcome,
            "duration_seconds": run.duration_seconds,
        },
        environment={"python": sys.version.split()[0], "isolation": "subprocess -I"},
        reproduction_steps=(
            f"bind variables: {dict(sorted(run.variables.items()))!r}",
            f"run with python -I -c (timeout {procedure.timeout_seconds}s):",
            procedure.program,
        ),
        related_concepts=(procedure.topic,),
    )


def reproduction_conflict(
    procedure: PracticeProcedure, first: PracticeRun, second: PracticeRun
) -> ConflictNode | None:
    """Identical variables must reproduce; a differing outcome is a conflict.

    Returns ``None`` when the runs agree (or aren't comparable).
    """
    if first.variables != second.variables:
        return None
    if first.outcome == second.outcome and first.status == second.status:
        return None
    return ConflictNode(
        concept=procedure.concept_name,
        interpretation_a=f"{run_claim(first)} [{first.id}]",
        interpretation_b=f"{run_claim(second)} [{second.id}]",
        sources=(first.id, second.id),
        evidence=(),  # provenance is the run ids; there is no raw document span
    )
