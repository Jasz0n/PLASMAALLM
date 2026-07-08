"""Repo-grounded practice (Roadmap M49): the repo's test suite is the grader.

M48's engine meets M49's ground truth. A repository test run is a
practice procedure like any other — executed, captured, content-
addressed — and a candidate patch is *tried in a disposable copy of the
repo*, never in the working tree. Ground truth by execution: the tests
pass or they don't.

Same isolation posture as the rest of the engine (subprocess + timeout;
OS-level isolation lands with M50): running a repo's tests executes that
repo's code, so point this only at trees you trust.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.kdp.types import content_hash
from allm.practice.executor import SandboxExecutor
from allm.practice.types import PracticeProcedure, PracticeRun, VariableSpec

logger = get_logger("practice.repo")

IGNORED = ("__pycache__", ".git", ".venv", ".pytest_cache", "node_modules")

_TEST_PROGRAM = """\
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "pytest", selector, "-q", "--tb=line", "-p", "no:cacheprovider"],
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    print("pass")
else:
    lines = [l for l in (result.stdout + result.stderr).splitlines() if l.strip()]
    print("fail: " + (lines[-1] if lines else "unknown"))
"""


def repo_test_procedure(
    repo_dir: Path | str, selector: str, *, timeout_seconds: float = 120.0
) -> PracticeProcedure:
    """The repo's own tests as a practice procedure (outcome: pass/fail)."""
    repo = Path(repo_dir)
    return PracticeProcedure(
        id=f"repo-test-{repo.name}-{content_hash(str(repo), selector)}",
        description=f"Run {selector!r} in repository {repo.name}; the tests are the grader.",
        program=_TEST_PROGRAM,
        variables=(VariableSpec(name="selector", default=selector),),
        topic=f"software-{repo.name.lower()}",
        timeout_seconds=timeout_seconds,
        workdir=str(repo),
    )


class CandidatePatch(BaseModel):
    """A proposed change: full new content for one repo-relative file.

    Full-file content (not a diff) keeps application deterministic and
    reviewable — what you read is exactly what lands.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    file: str
    content: str
    reasoning: str
    author: str

    @classmethod
    def build(cls, **kwargs: Any) -> "CandidatePatch":
        patch_id = "patch_" + content_hash(
            str(kwargs.get("file", "")),
            str(kwargs.get("content", "")),
            str(kwargs.get("author", "")),
        )
        return cls(id=patch_id, **kwargs)


def trial_patch(
    repo_dir: Path | str,
    patch: CandidatePatch,
    selector: str,
    *,
    executor: SandboxExecutor | None = None,
    timeout_seconds: float = 120.0,
) -> PracticeRun:
    """Try a patch in a disposable copy of the repo; never touch the original.

    Returns the test run from the patched copy — outcome ``pass`` means
    the candidate earned the right to become a contribution proposal.
    """
    repo = Path(repo_dir)
    target = Path(patch.file)
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"patch file must be repo-relative: {patch.file!r}")
    with tempfile.TemporaryDirectory(prefix="allm-trial-") as tmp:
        trial_repo = Path(tmp) / repo.name
        shutil.copytree(
            repo, trial_repo, ignore=shutil.ignore_patterns(*IGNORED)
        )
        destination = trial_repo / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(patch.content, encoding="utf-8")
        procedure = repo_test_procedure(trial_repo, selector, timeout_seconds=timeout_seconds)
        run = (executor or SandboxExecutor()).run(procedure)
    logger.info("trial %s on %s: %s", patch.id, repo.name, run.outcome[:60])
    return run
