"""Repo history as knowledge (Roadmap M49): commits and issues.

A merged commit is an **evidence package that already happened**: a
claim (the message), a measured change (the diffstat), a contributor
(the author) and an outcome (it landed on the main line). Ingesting
history therefore grows the evidence ledger — and KEL's Evidence Growth
Rate sees a repository's past exactly like fresh experiments.

Issues are the complementary signal: the project's own record of what
is broken, wanted or contested. They arrive as an exported JSONL file
(one issue per line) so the core stays offline; syncing that file from
a forge is the platform's job, not this module's. No network anywhere.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.evidence.types import EvidencePackage
from allm.researcher.types import KnowledgePackage

logger = get_logger("researcher.repo_history")

_LOG_FORMAT = "%H%x1f%an%x1f%aI%x1f%s"


class CommitRecord(BaseModel):
    """One commit, as recorded by the repository itself."""

    model_config = ConfigDict(frozen=True)

    sha: str
    author: str
    date: datetime
    subject: str
    files: tuple[str, ...] = ()
    insertions: int = 0
    deletions: int = 0


def git_commits(repo_dir: Path | str, *, limit: int = 100) -> list[CommitRecord]:
    """Read commit history via ``git log --numstat``; oldest first.

    Returns ``[]`` when the directory is not a git repository — history
    is optional knowledge, not a requirement.
    """
    repo = Path(repo_dir)
    try:
        completed = subprocess.run(
            [
                "git", "log", f"--max-count={limit}", "--numstat",
                f"--format=\x1e{_LOG_FORMAT}",
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("git log failed for %s: %s", repo, exc)
        return []
    if completed.returncode != 0:
        logger.info("%s is not a git repository (or git log failed)", repo)
        return []

    commits: list[CommitRecord] = []
    for block in completed.stdout.split("\x1e"):
        block = block.strip()
        if not block:
            continue
        header, _, body = block.partition("\n")
        parts = header.split("\x1f")
        if len(parts) != 4:
            continue
        sha, author, date, subject = parts
        files: list[str] = []
        insertions = deletions = 0
        for line in body.splitlines():
            columns = line.split("\t")
            if len(columns) != 3:
                continue
            added, removed, filename = columns
            files.append(filename)
            insertions += int(added) if added.isdigit() else 0
            deletions += int(removed) if removed.isdigit() else 0
        commits.append(
            CommitRecord(
                sha=sha,
                author=author,
                date=datetime.fromisoformat(date),
                subject=subject,
                files=tuple(files),
                insertions=insertions,
                deletions=deletions,
            )
        )
    commits.reverse()  # git log is newest-first; history reads oldest-first
    return commits


def commit_to_package(
    commit: CommitRecord, *, repo_name: str, concept: str | None = None
) -> EvidencePackage:
    """One merged commit → one evidence package that already happened."""
    return EvidencePackage.build(
        claim=f"[{repo_name}] {commit.subject}",
        concept=concept or f"software-{repo_name.lower()}",
        contributor=commit.author,
        kind="observation",  # we observed the landed change, not the work
        outcome="supported",  # it merged: the project accepted it
        measurements={
            "sha": commit.sha,
            "files_changed": len(commit.files),
            "insertions": commit.insertions,
            "deletions": commit.deletions,
        },
        reproduction_steps=(f"git show {commit.sha}",),
        related_concepts=tuple(sorted(set(commit.files))[:8]),
    )


def issues_from_jsonl(path: Path | str) -> list[dict]:
    """Load exported issues (one JSON object per line: title/body/state/labels)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def package_from_issues(
    path: Path | str, *, repo_name: str, provider: str
) -> KnowledgePackage:
    """Distill an issues export through KDP into one Knowledge Package.

    Open issues are the project's own statement of missing knowledge —
    they arrive as ``question`` units and feed curiosity, not confidence.
    """
    from allm.kdp.ingestion import DocumentStore
    from allm.kdp.pipeline import KDPipeline
    from allm.researcher.packages import package_from_distillation

    issues = issues_from_jsonl(path)
    if not issues:
        raise ValueError(f"no issues in {path}")
    store = DocumentStore()
    for issue in issues:
        state = issue.get("state", "open")
        labels = ", ".join(issue.get("labels", []))
        text = (
            f"Issue ({state}{', ' + labels if labels else ''}): "
            f"{issue.get('title', '')}\n\n{issue.get('body', '')}"
        )
        store.ingest_text(
            f"issue-{issue.get('id', issue.get('title', 'unknown'))}",
            text,
            context=f"repository:{repo_name}",
        )
    result = KDPipeline().distill(store)
    return package_from_distillation(
        result,
        provider=provider,
        title=f"Issues of {repo_name} ({len(issues)} exported)",
        provenance=str(path),
        curriculum_topic=f"software-{repo_name.lower()}",
    )
