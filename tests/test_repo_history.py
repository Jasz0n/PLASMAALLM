"""Repo history as evidence + issues ingestion (M49), offline."""

import subprocess
from pathlib import Path

import pytest

from allm.evidence import EvidenceLedger
from allm.kel.metrics import evidence_foundation
from allm.researcher.repo_history import (
    commit_to_package,
    git_commits,
    issues_from_jsonl,
    package_from_issues,
)
from allm.storage import SQLiteRecordStore

ISSUES = """\
{"id": 1, "title": "Widget cache never expires", "body": "The widget cache grows without bound. A widget cache entry is a rendered widget kept for reuse.", "state": "open", "labels": ["bug"]}
{"id": 2, "title": "Add dark theme", "body": "A theme is a named set of colors applied to every widget.", "state": "open", "labels": ["feature"]}
{"id": 3, "title": "Registry rejects duplicates", "body": "The registry is the central catalog of widget factories. Fixed by validating names.", "state": "closed", "labels": []}
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=dev", "-c", "user.email=dev@example.org", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "widgetlib"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "widget.py").write_text("WIDGETS = {}\n")
    _git(repo, "add", "widget.py")
    _git(repo, "commit", "-q", "-m", "add widget registry")
    (repo / "widget.py").write_text("WIDGETS = {}\nTHEMES = {}\n")
    (repo / "themes.py").write_text("DARK = 'dark'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "add theme support")
    return repo


def test_git_history_reads_oldest_first(git_repo: Path) -> None:
    commits = git_commits(git_repo)
    assert [c.subject for c in commits] == ["add widget registry", "add theme support"]
    assert commits[0].author == "dev"
    assert commits[1].files == ("themes.py", "widget.py")
    assert commits[1].insertions >= 2


def test_non_git_directory_yields_no_history(tmp_path: Path) -> None:
    assert git_commits(tmp_path) == []


def test_commits_become_evidence_and_grow_the_foundation(git_repo: Path) -> None:
    packages = [
        commit_to_package(c, repo_name="widgetlib") for c in git_commits(git_repo)
    ]
    assert all(p.kind == "observation" and p.outcome == "supported" for p in packages)
    assert packages[0].contributor == "dev"
    assert packages[0].measurements["files_changed"] == 1
    assert "git show" in packages[0].reproduction_steps[0]
    # the repository's past is measurable evidence: EGR's foundation sees it
    assert evidence_foundation(packages) > 0


def test_history_lands_in_the_ledger(tmp_path: Path, git_repo: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "ledger.sqlite3")
    try:
        ledger = EvidenceLedger(store)
        for commit in git_commits(git_repo):
            ledger.submit(commit_to_package(commit, repo_name="widgetlib"))
        packages = ledger.packages_for("software-widgetlib")
        assert len(packages) == 2
        breakdown = ledger.confidence("software-widgetlib")
        assert breakdown is not None and breakdown.contributors == 1
    finally:
        store.close()


def test_issues_distill_into_a_package(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    path.write_text(ISSUES)
    assert len(issues_from_jsonl(path)) == 3
    package = package_from_issues(path, repo_name="widgetlib", provider="repo-widgetlib")
    assert package.curriculum_topic == "software-widgetlib"
    assert package.provenance == str(path)
    assert package.concepts  # KDP extracted concepts from issue text
    names = " ".join(c.name.lower() for c in package.concepts)
    assert "theme" in names or "cache" in names or "registry" in names
