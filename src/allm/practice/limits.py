"""Kernel-enforced resource limits for executed code (Roadmap M50).

Defense-in-depth under the wall-clock timeout: the kernel — not our
grader — stops CPU spins, memory bombs and disk-filling, via POSIX
rlimits applied in the child process before ``exec``. On platforms
without the ``resource`` module (Windows) limits degrade to a logged
no-op; the timeout still applies. Full OS isolation (container/jail)
remains the M50 exit bar — rlimits are the floor, not the ceiling.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger

logger = get_logger("practice.limits")

try:  # POSIX only
    import resource as _resource
except ImportError:  # pragma: no cover - windows
    _resource = None


class ResourceLimits(BaseModel):
    """Per-execution kernel limits; ``None`` means inherit the parent's."""

    model_config = ConfigDict(frozen=True)

    cpu_seconds: int | None = Field(default=10, ge=1)
    memory_bytes: int | None = Field(default=512 * 1024 * 1024, ge=32 * 1024 * 1024)
    file_size_bytes: int | None = Field(default=8 * 1024 * 1024, ge=0)

    def preexec(self) -> Callable[[], None] | None:
        """The child-side hook for ``subprocess.run(preexec_fn=...)``.

        Returns ``None`` when the platform cannot enforce limits, so
        callers degrade gracefully instead of failing to execute.
        """
        if _resource is None:
            logger.warning("resource limits unavailable on this platform")
            return None
        cpu, memory, fsize = self.cpu_seconds, self.memory_bytes, self.file_size_bytes

        def apply() -> None:  # pragma: no cover - runs in the child process
            _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
            if cpu is not None:
                _resource.setrlimit(_resource.RLIMIT_CPU, (cpu, cpu))
            if memory is not None:
                _resource.setrlimit(_resource.RLIMIT_AS, (memory, memory))
            if fsize is not None:
                _resource.setrlimit(_resource.RLIMIT_FSIZE, (fsize, fsize))

        return apply


# Repo trials spawn pytest over a whole test suite: same kernel guard,
# roomier budget.
REPO_TASK_LIMITS = ResourceLimits(
    cpu_seconds=300,
    memory_bytes=2 * 1024 * 1024 * 1024,
    file_size_bytes=64 * 1024 * 1024,
)
