"""CLI test for `allm kdp distill`."""

from pathlib import Path

import pytest

from allm.cli.main import main


def test_kdp_distill_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    doc = tmp_path / "lecture.md"
    doc.write_text(
        "Self-attention is a mechanism relating tokens to each other.\n\n"
        "People often think attention reads words one by one, but it does not.\n",
        encoding="utf-8",
    )
    db = tmp_path / "kdp.sqlite3"
    assert main(["kdp", "distill", str(doc), "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "[concept] Self-Attention" in out
    assert "graph: " in out
    assert db.exists()


def test_kdp_distill_without_db(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    doc = tmp_path / "note.md"
    doc.write_text("Gradient descent is an optimisation algorithm.", encoding="utf-8")
    assert main(["kdp", "distill", str(doc)]) == 0
    assert "1 document(s)" in capsys.readouterr().out
