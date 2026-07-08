"""Practice Engine value objects (see PRACTICE.md).

A :class:`PracticeProcedure` is a parameterized pure-Python program
whose stdout is its observable outcome; a :class:`PracticeRun` is the
ground-truth record of one execution. Run ids are content-addressed
from (procedure, variables, outcome): the same experiment always names
the same record.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.kdp.types import content_hash
from allm.practice.limits import ResourceLimits

RunStatus = Literal["ok", "crash", "timeout"]


class VariableSpec(BaseModel):
    """One tunable of a procedure: its default and candidate values."""

    model_config = ConfigDict(frozen=True)

    name: str
    default: Any
    candidates: tuple[Any, ...] = ()
    description: str = ""


class PracticeProcedure(BaseModel):
    """A parameterized program; its stdout is the observable outcome.

    ``program`` is pure Python (stdlib only) that reads its variables
    as ordinary names — the executor binds them as literals, it never
    interpolates values into code text.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    program: str
    variables: tuple[VariableSpec, ...]
    topic: str = "practice"
    timeout_seconds: float = 5.0
    # Repo-grounded procedures (M49) run inside a working tree; None
    # keeps the classic behaviour (current process cwd).
    workdir: str | None = None
    # Kernel-enforced budget (M50): CPU, memory, file size.
    limits: ResourceLimits = ResourceLimits()

    @property
    def concept_name(self) -> str:
        """The knowledge-graph concept this procedure's outcomes attach to."""
        return f"practice:{self.id}"

    def defaults(self) -> dict[str, Any]:
        return {v.name: v.default for v in self.variables}

    def variable(self, name: str) -> VariableSpec:
        for spec in self.variables:
            if spec.name == name:
                return spec
        raise KeyError(f"procedure {self.id!r} has no variable {name!r}")


class PracticeRun(BaseModel):
    """Ground truth: what one execution actually did."""

    model_config = ConfigDict(frozen=True)

    id: str
    procedure_id: str
    variables: dict[str, Any]
    status: RunStatus
    outcome: str
    stdout: str
    stderr: str
    duration_seconds: float
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def build(cls, **kwargs: Any) -> "PracticeRun":
        """Construct with a deterministic content-derived id."""
        run_id = "run_" + content_hash(
            str(kwargs.get("procedure_id", "")),
            str(sorted(kwargs.get("variables", {}).items())),
            str(kwargs.get("status", "")),
            str(kwargs.get("outcome", "")),
        )
        return cls(id=run_id, **kwargs)


class SweepResult(BaseModel):
    """One variable varied, the rest fixed: does the outcome depend on it?"""

    model_config = ConfigDict(frozen=True)

    procedure_id: str
    variable: str
    fixed: dict[str, Any]
    runs: tuple[PracticeRun, ...]
    depends: bool

    @property
    def relation(self) -> str:
        """The graph relation this sweep earned."""
        kind = "outcome-depends-on" if self.depends else "outcome-independent-of"
        return f"{kind}:{self.variable}"
