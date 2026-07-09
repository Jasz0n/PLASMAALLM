"""allm seed — the public-loop rehearsal (M52), offline."""

from pathlib import Path

from allm.cli.main import main
from allm.seed import seed_public_loop
from allm.storage import SQLiteRecordStore


def test_seed_runs_the_whole_public_loop(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "seed.sqlite3")
    try:
        report = seed_public_loop(store)
    finally:
        store.close()

    # discussion → concepts, including the contested one
    assert "The Nano Coating" in report.concepts
    assert report.contested_concept == "The Nano Coating"

    # conflict → proposal → replications → confidence shift (only evidence moves it)
    assert report.proposal_outcome == "supported"
    assert report.confidence_after > report.confidence_before
    assert report.events >= 8  # the loop is visible on the feed


def test_seeded_store_makes_the_dashboard_alive(tmp_path: Path) -> None:
    from allm.api.dashboard import system_state

    store = SQLiteRecordStore(tmp_path / "seed.sqlite3")
    try:
        seed_public_loop(store)
        state = system_state(store)
    finally:
        store.close()

    assert state["proposals"]["by_status"].get("resolved") == 1
    assert state["evidence"]["replications"] >= 2
    # the KEL scorecard has real measurements, not blanks
    assert any(m["latest"] is not None for m in state["kel"]["metrics"])
    assert state["events"]["total"] >= 8


def test_cli_seed_refuses_to_clobber_without_force(tmp_path: Path, capsys) -> None:
    db = str(tmp_path / "seed.sqlite3")
    assert main(["seed", "--db", db]) == 0
    assert "contested" in capsys.readouterr().out

    # a second run must not silently double-seed
    assert main(["seed", "--db", db]) == 1
    assert "already has data" in capsys.readouterr().out

    assert main(["seed", "--db", db, "--force"]) == 0
