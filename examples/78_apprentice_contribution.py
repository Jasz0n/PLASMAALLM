"""M49: the apprentice loop — study, fix, trial, propose, human approves.

A repository has a failing test. The student reads the buggy file and
the test output, proposes a fix as a candidate patch, and the patch is
tried in a *disposable copy* of the repo — the repo's own test suite is
the grader. Only then does it become a contribution proposal, and only
a named human approval unlocks applying it. Nothing is ever pushed;
what happens after the file lands is a human's git history.

    PYTHONPATH=src python3 examples/78_apprentice_contribution.py
    ALLM_PRACTICE_STUDENT=echo ...   # offline: shows the rejection path
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

from allm.core.logging import setup_logging
from allm.exam.coding import extract_code
from allm.models import EchoModel, ModelSpec
from allm.models.base import model_loaders
from allm.practice import (
    ApprovalError,
    CandidatePatch,
    ContributionBoard,
    SandboxExecutor,
    record_review_outcome,
    repo_test_procedure,
    trial_patch,
)
from allm.storage import SQLiteRecordStore
from allm.students import FailureLog
from allm.teacher import KnowledgeState

BUGGY = '''"""Tiny pricing helpers for the demo shop."""


def line_total(price: float, quantity: int, discount: float = 0.0) -> float:
    """Total for one order line; ``discount`` is a fraction like 0.1."""
    return round(price * quantity * (1 + discount), 2)
'''

TEST = '''from pricing import line_total


def test_plain_total():
    assert line_total(10.0, 3) == 30.0


def test_discount_reduces_total():
    assert line_total(10.0, 3, discount=0.1) == 27.0
'''


def pick_model():
    if os.environ.get("ALLM_PRACTICE_STUDENT") != "echo":
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
            model_id = os.environ.get("ALLM_STUDENT_MODEL", "qwen2.5:7b-instruct")
            spec = ModelSpec(name="apprentice", provider="ollama", model_id=model_id)
            return model_loaders.get("ollama")().load(spec), model_id
        except Exception:
            print("(Ollama unreachable — echo fallback shows the rejection path)")
    return EchoModel(ModelSpec(name="apprentice", provider="echo", model_id="none")), "echo"


def main() -> None:
    setup_logging("WARNING")
    workdir = Path(tempfile.mkdtemp(prefix="allm-apprentice-"))
    repo = workdir / "shoplib"
    repo.mkdir()
    (repo / "pricing.py").write_text(BUGGY)
    (repo / "test_pricing.py").write_text(TEST)
    executor = SandboxExecutor()

    print("=== 1. The repo's tests are the grader ===")
    before = executor.run(repo_test_procedure(repo, "test_pricing.py"))
    print(f"working tree: {before.outcome}")

    model, model_id = pick_model()
    print(f"\n=== 2. Apprentice ({model_id}) proposes a fix ===")
    prompt = (
        "You are a software apprentice fixing a bug.\n"
        f"File pricing.py:\n```python\n{BUGGY}```\n"
        f"Failing tests (test_pricing.py):\n```python\n{TEST}```\n"
        f"Test output: {before.outcome}\n"
        "Reply with ONLY the complete corrected content of pricing.py "
        "in one python code block. Keep the docstrings."
    )
    patch = CandidatePatch.build(
        file="pricing.py",
        content=extract_code(model.generate(prompt)).rstrip() + "\n",
        reasoning=f"proposed by {model_id} from the failing test output",
        author=f"apprentice-{model_id}",
    )

    print("=== 3. Trial in a disposable copy (working tree untouched) ===")
    trial = trial_patch(repo, patch, "test_pricing.py", executor=executor)
    print(f"trial verdict: {trial.outcome}")
    assert (repo / "pricing.py").read_text() == BUGGY

    store = SQLiteRecordStore(workdir / "contrib.sqlite3")
    board = ContributionBoard(store)
    contribution = board.propose(patch, test_selector="test_pricing.py", trial=trial)
    print(f"\n=== 4. Contribution {contribution.id} [{contribution.status}] ===")

    print("\n=== 5. The invariant: applying without approval must fail ===")
    try:
        board.apply(contribution.id, repo)
    except ApprovalError as exc:
        print(f"blocked as designed: {exc}")

    print("\n=== 6. A named human reviews ===")
    reviewer = os.environ.get("ALLM_REVIEWER", "maintainer")
    if trial.outcome == "pass":
        reviewed = board.approve(
            contribution.id, reviewer=reviewer, reason="trial passed; change is minimal"
        )
        target = board.apply(contribution.id, repo)
        after = executor.run(repo_test_procedure(repo, "test_pricing.py"))
        print(f"approved by {reviewer}; applied to {target.name}; repo tests: {after.outcome}")
    else:
        reviewed = board.reject(
            contribution.id, reviewer=reviewer, reason=f"trial failed: {trial.outcome[:60]}"
        )
        print(f"rejected by {reviewer} [{reviewed.status}] — the reason becomes a failure sample")

    print("\n=== 7. The verdict becomes learning signal (M49 outcome feedback) ===")
    state = KnowledgeState(store)
    failures = FailureLog(store)
    result = record_review_outcome(
        reviewed, state=state, failures=failures, topic="software-shoplib"
    )
    author = reviewed.patch.author
    print(f"review exam score: {result.score:.2f}")
    print(f"{author} confidence on 'software-shoplib': "
          f"{state.confidence(author, 'software-shoplib'):.2f}")
    for failure in failures.failures(author):
        print(f"failure logged with the reviewer's reason: {failure.feedback!r}")

    store.close()
    print(f"\nworkdir: {workdir}")


if __name__ == "__main__":
    main()
