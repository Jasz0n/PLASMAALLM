"""Repo-grounded practice + contribution approval invariant (M49), offline."""

from pathlib import Path

import pytest

from allm.practice import (
    ApprovalError,
    CandidatePatch,
    ContributionBoard,
    SandboxExecutor,
    repo_test_procedure,
    trial_patch,
)
from allm.storage import SQLiteRecordStore

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "calcrepo"
    root.mkdir()
    (root / "calculator.py").write_text(BUGGY)
    (root / "test_calculator.py").write_text(TEST)
    return root


@pytest.fixture()
def executor() -> SandboxExecutor:
    return SandboxExecutor()


def test_repo_tests_are_the_grader(repo: Path, executor: SandboxExecutor) -> None:
    run = executor.run(repo_test_procedure(repo, "test_calculator.py"))
    assert run.status == "ok"
    assert run.outcome.startswith("fail")  # the bug is real


def test_trial_patch_never_touches_the_working_tree(
    repo: Path, executor: SandboxExecutor
) -> None:
    patch = CandidatePatch.build(
        file="calculator.py",
        content=FIXED,
        reasoning="add() subtracted; tests expect addition",
        author="student-alpha",
    )
    run = trial_patch(repo, patch, "test_calculator.py", executor=executor)
    assert run.outcome == "pass"
    assert (repo / "calculator.py").read_text() == BUGGY  # untouched


def test_patch_paths_must_stay_inside_the_repo() -> None:
    with pytest.raises(ValueError, match="repo-relative"):
        trial_patch(
            Path("."),
            CandidatePatch.build(
                file="../escape.py", content="x", reasoning="r", author="a"
            ),
            "test_calculator.py",
        )


def test_nothing_leaves_without_human_approval(
    tmp_path: Path, repo: Path, executor: SandboxExecutor
) -> None:
    patch = CandidatePatch.build(
        file="calculator.py", content=FIXED, reasoning="fix add", author="student-alpha"
    )
    trial = trial_patch(repo, patch, "test_calculator.py", executor=executor)
    store = SQLiteRecordStore(tmp_path / "contrib.sqlite3")
    try:
        board = ContributionBoard(store)
        contribution = board.propose(patch, test_selector="test_calculator.py", trial=trial)
        assert contribution.status == "proposed"

        # THE invariant: applying without an approval record is impossible
        with pytest.raises(ApprovalError, match="human approval record"):
            board.apply(contribution.id, repo)
        assert (repo / "calculator.py").read_text() == BUGGY

        # approval requires a named human and a reason
        with pytest.raises(ApprovalError, match="named human"):
            board.approve(contribution.id, reviewer="  ", reason="ok")
        with pytest.raises(ApprovalError, match="reason"):
            board.approve(contribution.id, reviewer="maintainer", reason="")

        approved = board.approve(
            contribution.id, reviewer="maintainer", reason="correct and minimal"
        )
        assert approved.reviewer == "maintainer"

        target = board.apply(contribution.id, repo)
        assert target.read_text() == FIXED
        # the repo's own tests confirm the applied contribution
        assert executor.run(repo_test_procedure(repo, "test_calculator.py")).outcome == "pass"
        assert board.get(contribution.id).status == "applied"
    finally:
        store.close()


def test_rejection_needs_reviewer_and_reason_too(
    tmp_path: Path, repo: Path, executor: SandboxExecutor
) -> None:
    patch = CandidatePatch.build(
        file="calculator.py", content="def add(a, b):\n    return 5\n",
        reasoning="hardcode", author="student-beta",
    )
    trial = trial_patch(repo, patch, "test_calculator.py", executor=executor)
    assert trial.outcome == "pass"  # passes the test, still worth rejecting
    store = SQLiteRecordStore(tmp_path / "contrib.sqlite3")
    try:
        board = ContributionBoard(store)
        contribution = board.propose(patch, test_selector="test_calculator.py", trial=trial)
        rejected = board.reject(
            contribution.id, reviewer="maintainer", reason="hardcoded value, not a fix"
        )
        assert rejected.status == "rejected"
        with pytest.raises(ApprovalError):
            board.apply(contribution.id, repo)
        # a settled review cannot be re-reviewed silently
        with pytest.raises(ApprovalError, match="already"):
            board.approve(contribution.id, reviewer="other", reason="changed my mind")
    finally:
        store.close()


def test_no_remote_operations_exist() -> None:
    """The invariant, structurally: no push/remote code in the module."""
    import allm.practice.contribution as contribution_module

    source = Path(contribution_module.__file__).read_text()
    for forbidden in ("git push", "subprocess", "requests", "urllib"):
        assert forbidden not in source


def _reviewed_contribution(tmp_path, repo, executor, *, approve: bool):
    from allm.practice import record_review_outcome
    from allm.students import FailureLog
    from allm.teacher import KnowledgeState

    content = FIXED if approve else "def add(a, b):\n    return 5\n"
    patch = CandidatePatch.build(
        file="calculator.py", content=content,
        reasoning="fix add" if approve else "hardcode",
        author="apprentice",
    )
    trial = trial_patch(repo, patch, "test_calculator.py", executor=executor)
    store = SQLiteRecordStore(tmp_path / "feedback.sqlite3")
    board = ContributionBoard(store)
    contribution = board.propose(patch, test_selector="test_calculator.py", trial=trial)
    if approve:
        contribution = board.approve(
            contribution.id, reviewer="maintainer", reason="correct and minimal"
        )
    else:
        contribution = board.reject(
            contribution.id, reviewer="maintainer", reason="hardcoded value, not a fix"
        )
    state = KnowledgeState(store)
    failures = FailureLog(store)
    result = record_review_outcome(
        contribution, state=state, failures=failures, topic="software-calcrepo"
    )
    return store, state, failures, result


def test_approval_raises_topic_confidence(tmp_path, repo, executor) -> None:
    store, state, failures, result = _reviewed_contribution(
        tmp_path, repo, executor, approve=True
    )
    try:
        assert result.score == 1.0
        assert state.confidence("apprentice", "software-calcrepo") == 1.0
        assert failures.failures("apprentice") == []
        # the approved patch is now a studyable expected answer
        assert result.results[0].question.expected == FIXED
    finally:
        store.close()


def test_rejection_becomes_failure_with_the_reason(tmp_path, repo, executor) -> None:
    store, state, failures, result = _reviewed_contribution(
        tmp_path, repo, executor, approve=False
    )
    try:
        assert result.score == 0.0
        assert state.confidence("apprentice", "software-calcrepo") == 0.0
        (failure,) = failures.failures("apprentice")
        assert failure.feedback == "hardcoded value, not a fix"
        assert failure.expected is None  # the right answer is still unknown
    finally:
        store.close()


def test_unreviewed_contribution_has_no_signal(tmp_path, repo, executor) -> None:
    from allm.practice import review_exam_result

    patch = CandidatePatch.build(
        file="calculator.py", content=FIXED, reasoning="fix", author="apprentice"
    )
    trial = trial_patch(repo, patch, "test_calculator.py", executor=executor)
    store = SQLiteRecordStore(tmp_path / "feedback.sqlite3")
    try:
        board = ContributionBoard(store)
        contribution = board.propose(patch, test_selector="test_calculator.py", trial=trial)
        with pytest.raises(ValueError, match="only reviewed"):
            review_exam_result(contribution, topic="software-calcrepo")
    finally:
        store.close()
