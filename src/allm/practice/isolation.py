"""Container-grade execution isolation via bubblewrap (Roadmap M50).

The M50 exit bar made real: when ``bwrap`` is available, every executed
procedure runs in its own mount + network + PID namespace — root
filesystem read-only, ``/tmp`` a private tmpfs, **no network at all**
(PRACTICE.md's "no network inside procedures" stops being policy and
becomes kernel enforcement). Only the procedure's declared workdir is
writable.

Layered under the existing guards, never instead of them: wall-clock
timeout and POSIX rlimits (``limits.py``) still apply inside the
namespace. Where bubblewrap is missing the executor degrades to
subprocess + rlimits with a logged warning — never silently.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from typing import Literal

from allm.core.logging import get_logger

logger = get_logger("practice.isolation")

IsolationMode = Literal["auto", "bwrap", "none"]

_PROFILE = (
    "--ro-bind", "/", "/",
    "--dev", "/dev",
    "--proc", "/proc",
    "--tmpfs", "/tmp",
    "--unshare-all",
    "--die-with-parent",
)


@functools.lru_cache(maxsize=1)
def bwrap_available() -> bool:
    """Probe once whether bubblewrap actually works here.

    Installed is not enough — user namespaces can be disabled in
    containers/VMs, so we run a real echo through the full profile.
    """
    if shutil.which("bwrap") is None:
        return False
    try:
        completed = subprocess.run(
            ["bwrap", *_PROFILE, "--", "/bin/echo", "ok"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    working = completed.returncode == 0 and completed.stdout.strip() == "ok"
    if not working:
        logger.warning("bwrap installed but not functional: %s", completed.stderr.strip()[:120])
    return working


def resolve_isolation(mode: IsolationMode) -> Literal["bwrap", "none"]:
    """Resolve ``auto`` against reality; fail loudly on impossible asks."""
    if mode == "bwrap":
        if not bwrap_available():
            raise RuntimeError("isolation 'bwrap' requested but bubblewrap is not functional")
        return "bwrap"
    if mode == "auto":
        if bwrap_available():
            return "bwrap"
        logger.warning(
            "bubblewrap unavailable — executing with subprocess + rlimits only "
            "(install bubblewrap for namespace isolation)"
        )
        return "none"
    return "none"


def wrap_command(
    argv: list[str], *, workdir: str | None = None
) -> list[str]:
    """Wrap ``argv`` in the bubblewrap profile.

    The declared workdir (a disposable trial copy for repo tasks) is the
    only writable path besides the private ``/tmp``.
    """
    command = ["bwrap", *_PROFILE]
    if workdir is not None:
        command += ["--bind", workdir, workdir, "--chdir", workdir]
    command += ["--", *argv]
    return command
