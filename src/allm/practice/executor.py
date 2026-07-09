"""Sandboxed procedure execution (PRACTICE.md Stage 3).

Three guard layers, outermost first (Roadmap M50):

1. **Namespace isolation** (bubblewrap, when available): read-only
   root, private /tmp, no network, only the declared workdir writable.
2. **Kernel rlimits**: CPU, memory, file size — the kernel stops what
   a timeout cannot.
3. **Wall-clock timeout** + ``python -I`` + literal variable binding —
   values are never interpolated into code text.

Crashes and timeouts are outcomes, not errors: the run records what
actually happened.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

from allm.core.logging import get_logger
from allm.practice.isolation import IsolationMode, resolve_isolation, wrap_command
from allm.practice.limits import clean_env
from allm.practice.types import PracticeProcedure, PracticeRun

logger = get_logger("practice.executor")

_LITERALS = (str, int, float, bool, bytes, type(None))


def bind_variables(variables: dict[str, Any]) -> str:
    """Render variables as a literal-assignment prelude.

    Only plain literals (and tuples/lists/dicts of them) are allowed;
    anything else could smuggle code into the sandbox.
    """
    lines = []
    for name, value in sorted(variables.items()):
        if not name.isidentifier():
            raise ValueError(f"invalid variable name {name!r}")
        _check_literal(name, value)
        lines.append(f"{name} = {value!r}")
    return "\n".join(lines)


def _check_literal(name: str, value: Any) -> None:
    if isinstance(value, _LITERALS):
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _check_literal(name, item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _check_literal(name, key)
            _check_literal(name, item)
        return
    raise ValueError(f"variable {name!r} is not a plain literal: {type(value).__name__}")


class SandboxExecutor:
    """Runs procedures behind namespace isolation, rlimits and a timeout.

    ``isolation``: ``auto`` (default — bubblewrap when functional,
    logged fallback otherwise), ``bwrap`` (required, raises when
    impossible) or ``none``. Overridable via ``ALLM_SANDBOX_ISOLATION``.
    """

    def __init__(self, isolation: IsolationMode | None = None) -> None:
        requested = isolation or os.environ.get("ALLM_SANDBOX_ISOLATION", "auto")
        if requested not in ("auto", "bwrap", "none"):
            raise ValueError(f"unknown isolation mode {requested!r}")
        self._isolation = resolve_isolation(requested)

    @property
    def isolation(self) -> str:
        """The resolved isolation actually in effect (``bwrap``/``none``)."""
        return self._isolation

    def run(
        self, procedure: PracticeProcedure, variables: dict[str, Any] | None = None
    ) -> PracticeRun:
        """Execute once with ``variables`` (defaults fill the gaps)."""
        bound = {**procedure.defaults(), **(variables or {})}
        unknown = set(bound) - {v.name for v in procedure.variables}
        if unknown:
            raise ValueError(f"unknown variable(s) for {procedure.id!r}: {sorted(unknown)}")
        code = bind_variables(bound) + "\n" + procedure.program
        argv = [sys.executable, "-I", "-c", code]
        if self._isolation == "bwrap":
            argv = wrap_command(argv, workdir=procedure.workdir)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=procedure.timeout_seconds,
                cwd=procedure.workdir,
                preexec_fn=procedure.limits.preexec(),
                env=clean_env(),
            )
            duration = time.perf_counter() - started
            status = "ok" if completed.returncode == 0 else "crash"
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - started
            status = "timeout"
            stdout = exc.stdout or "" if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr or "" if isinstance(exc.stderr, str) else ""
        outcome = _outcome(status, stdout, stderr)
        run = PracticeRun.build(
            procedure_id=procedure.id,
            variables=bound,
            status=status,
            outcome=outcome,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(duration, 4),
        )
        logger.info(
            "%s %s -> %s (%r)", procedure.id, bound, status, outcome[:60]
        )
        return run


def _outcome(status: str, stdout: str, stderr: str) -> str:
    """The observable result: stdout when ok, the failure signature otherwise."""
    if status == "ok":
        return stdout.strip()
    if status == "timeout":
        return "timeout"
    error_lines = stderr.strip().splitlines()
    return f"crash: {error_lines[-1] if error_lines else 'unknown'}"
