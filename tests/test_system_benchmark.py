"""State-of-the-system benchmark (Roadmap M47), offline."""

from pathlib import Path

import pytest

from allm.benchmarks import run_system_benchmark
from allm.benchmarks.system_report import fiction_samples, split_samples
from allm.cli.main import main

ROOT = Path(__file__).resolve().parents[1]


def test_split_is_deterministic_and_disjoint() -> None:
    samples = fiction_samples()
    train, holdout = split_samples(samples)
    train2, holdout2 = split_samples(list(reversed(samples)))
    assert [s.id for s in train] == [s.id for s in train2]
    assert [s.id for s in holdout] == [s.id for s in holdout2]
    assert holdout
    assert not {s.id for s in train} & {s.id for s in holdout}


def test_fiction_benchmark_shows_learning(tmp_path: Path) -> None:
    report = run_system_benchmark(
        ("fiction",), iterations=2, seed=13, root=ROOT, workdir=tmp_path
    )
    (row,) = report.corpora
    assert row.corpus == "fiction"
    assert row.samples_train > 0 and row.samples_holdout > 0
    assert row.score_after > row.score_before
    assert row.learning_gain is not None and row.learning_gain > 0
    # the echo student memorises; it must not appear to generalize
    assert row.holdout_gap == pytest.approx(row.score_after - row.holdout_score)
    assert "| fiction |" in report.to_markdown()


MINI_BOOK = """
Plasma is a field of energy in constant motion. Plasma is a field of
energy in constant motion around a central point. The magnetic field is
the flow of energy from a plasma. The magnetic field is energy flowing
outward from a plasma. Matter is the interaction of gravitational and
magnetic fields. Matter is created where gravitational and magnetic
fields interact. The gravitational field pulls energy inward toward the
plasma center.
"""


def test_books_benchmark_reports_rcr(tmp_path: Path) -> None:
    # A synthetic mini book keeps the offline suite fast; the real
    # three-book corpus runs through the same path via `allm benchmark`.
    (tmp_path / "books").mkdir()
    (tmp_path / "books" / "mini_book.txt").write_text(MINI_BOOK)
    report = run_system_benchmark(
        ("books",), iterations=1, seed=13, limit=12, root=tmp_path, workdir=tmp_path
    )
    (row,) = report.corpora
    assert row.samples_train > 0
    assert row.rcr is not None


def test_practice_corpus_earns_evidence(tmp_path: Path) -> None:
    report = run_system_benchmark(
        ("practice",), iterations=1, seed=13, limit=0, root=ROOT, workdir=tmp_path
    )
    (row,) = report.corpora
    assert row.samples_train > 0
    # the run *earned* its evidence: packages exist for every sample
    assert row.evidence_growth is not None and row.evidence_growth > 0
    # text corpora produce no packages, so their EGR must honestly read 0
    text = run_system_benchmark(
        ("fiction",), iterations=1, seed=13, root=ROOT, workdir=tmp_path
    )
    assert text.corpora[0].evidence_growth == 0.0


def test_cli_benchmark_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    code = main(
        [
            "--log-level",
            "WARNING",
            "benchmark",
            "--corpora",
            "fiction",
            "--iterations",
            "1",
            "--root",
            str(ROOT),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert output.exists() and '"corpus": "fiction"' in output.read_text()


def test_unknown_corpus_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown corpus"):
        run_system_benchmark(("nope",), iterations=1, root=ROOT, workdir=tmp_path)


def test_missing_corpus_is_skipped_not_fatal(tmp_path: Path) -> None:
    # empty root: no books sidecars, no kids corpus
    report = run_system_benchmark(
        ("fiction", "books"), iterations=1, seed=13, root=tmp_path, workdir=tmp_path
    )
    assert [row.corpus for row in report.corpora] == ["fiction"]
    (note,) = report.skipped
    assert note.startswith("books")
    assert "skipped: books" in report.to_markdown()
