"""M49: a repository's past is evidence that already happened.

Every merged commit is a claim + a measured change + a contributor + an
accepted outcome — an evidence package the project produced before ALLM
ever arrived. This demo builds a small git repo, ingests its history
into the evidence ledger, distills its exported issues through KDP, and
lets KEL measure the result: EGR sees a repository's past exactly like
fresh experiments.

    PYTHONPATH=src python3 examples/79_repo_history_evidence.py
    ALLM_REPO_DIR=/path/to/git/repo ...   # use a real repository instead
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.evidence import EvidenceLedger
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.researcher import inject_package_concepts
from allm.researcher.repo_history import (
    commit_to_package,
    git_commits,
    package_from_issues,
)
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState

ISSUES = """\
{"id": 1, "title": "Cache never expires", "body": "The render cache grows without bound; a cache entry is a rendered widget kept for reuse.", "state": "open", "labels": ["bug"]}
{"id": 2, "title": "Add dark theme", "body": "A theme is a named set of colors applied to every widget.", "state": "open", "labels": ["feature"]}
"""


def build_demo_repo(root: Path) -> Path:
    repo = root / "widgetlib"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=ada", "-c", "user.email=ada@example.org", *args],
            cwd=repo, check=True, capture_output=True,
        )

    git("init", "-q")
    for name, content, message in (
        ("widget.py", "WIDGETS = {}\n", "add widget registry"),
        ("cache.py", "CACHE = {}\n", "add render cache"),
        ("themes.py", "DARK = 'dark'\n", "add theme scaffolding"),
    ):
        (repo / name).write_text(content)
        git("add", name)
        git("commit", "-q", "-m", message)
    return repo


def main() -> None:
    setup_logging("WARNING")
    workdir = Path(tempfile.mkdtemp(prefix="allm-repo-history-"))
    repo = Path(os.environ["ALLM_REPO_DIR"]) if os.environ.get("ALLM_REPO_DIR") else build_demo_repo(workdir)
    store = SQLiteRecordStore(workdir / "history.sqlite3")
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    kel = KnowledgeEvaluationLayer(graph, store, KnowledgeState(store), ledger=ledger)
    kel.evaluate()  # baseline: EGR measures what ingestion earns

    print(f"=== 1. Commits of {repo.name} become evidence packages ===")
    commits = git_commits(repo, limit=int(os.environ.get("ALLM_REPO_COMMITS", "50")))
    for commit in commits:
        package = commit_to_package(commit, repo_name=repo.name)
        ledger.submit(package)
    print(f"{len(commits)} commit(s) ingested; example claim:")
    if commits:
        example = commit_to_package(commits[0], repo_name=repo.name)
        print(f"  {example.claim!r} by {example.contributor}")
        print(f"  measurements: {example.measurements}")

    print("\n=== 2. Exported issues distill through KDP ===")
    issues_path = workdir / "issues.jsonl"
    issues_path.write_text(ISSUES)
    package = package_from_issues(issues_path, repo_name=repo.name, provider=f"repo-{repo.name}")
    counts = inject_package_concepts(graph, package)
    print(f"{package.title}: {len(package.concepts)} concept(s), "
          f"{counts['added']} added to the graph")
    for concept in package.concepts[:5]:
        print(f"  - {concept.name}: {concept.description[:60]}")

    print("\n=== 3. KEL measures the ingested past ===")
    report = kel.evaluate()
    breakdown = ledger.confidence(f"software-{repo.name.lower()}")
    print(f"EGR (evidence growth): {report.egr}")
    if breakdown:
        print(f"confidence on 'software-{repo.name.lower()}': {breakdown.value:.2f} "
              f"from {breakdown.contributors} contributor(s), "
              f"{len(breakdown.packages)} package(s)")

    store.close()
    print(f"\nworkdir: {workdir}")


if __name__ == "__main__":
    main()
