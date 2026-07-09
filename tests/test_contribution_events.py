"""Contribution lifecycle on the live feed (M51): the approval ledger, streamed."""

from pathlib import Path

import pytest

from allm.events import EventLog
from allm.practice.contribution import ApprovalError, ContributionBoard
from allm.practice.repo_tasks import CandidatePatch
from allm.practice.types import PracticeRun
from allm.storage import SQLiteRecordStore


def _trial() -> PracticeRun:
    return PracticeRun.build(
        procedure_id="repo-test-x",
        variables={},
        status="ok",
        outcome="pass",
        stdout="pass",
        stderr="",
        duration_seconds=0.1,
    )


@pytest.fixture()
def board_and_log(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "contrib.sqlite3")
    log = EventLog(store)
    return ContributionBoard(store, events=log), log


def test_each_transition_emits_a_contribution_event(board_and_log) -> None:
    board, log = board_and_log
    patch = CandidatePatch.build(
        file="src/fix.py", content="x = 1\n", reasoning="fix", author="apprentice"
    )
    contribution = board.propose(patch, test_selector="tests/test_fix.py", trial=_trial())
    board.approve(contribution.id, reviewer="jasz0n", reason="correct and minimal")

    types = [e.type for e in log.since(0)]
    assert types == ["contribution.proposed", "contribution.approved"]
    approved = log.since(0)[-1]
    assert approved.subject == contribution.id
    assert approved.data["author"] == "apprentice"
    assert approved.data["reviewer"] == "jasz0n"


def test_apply_emits_and_rejection_is_its_own_event(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "c.sqlite3")
    log = EventLog(store)
    board = ContributionBoard(store, events=log)

    # rejection path
    p1 = CandidatePatch.build(file="a.py", content="1\n", reasoning="r", author="bot")
    c1 = board.propose(p1, test_selector="t", trial=_trial())
    board.reject(c1.id, reviewer="ada", reason="breaks another test")

    # approve + apply path
    p2 = CandidatePatch.build(file="b.py", content="2\n", reasoning="r", author="bot")
    c2 = board.propose(p2, test_selector="t", trial=_trial())
    board.approve(c2.id, reviewer="ada", reason="good")
    board.apply(c2.id, tmp_path / "repo")

    assert [e.type for e in log.since(0)] == [
        "contribution.proposed",
        "contribution.rejected",
        "contribution.proposed",
        "contribution.approved",
        "contribution.applied",
    ]


def test_board_without_a_log_still_works(tmp_path: Path) -> None:
    # the event log is optional; the invariant is unchanged without it
    board = ContributionBoard(SQLiteRecordStore(tmp_path / "c.sqlite3"))
    patch = CandidatePatch.build(file="a.py", content="1\n", reasoning="r", author="bot")
    contribution = board.propose(patch, test_selector="t", trial=_trial())
    with pytest.raises(ApprovalError):
        board.apply(contribution.id, tmp_path / "repo")  # still gated
