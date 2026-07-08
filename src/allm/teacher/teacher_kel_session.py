"""Teacher review pause/resume session for KEL loops (M34)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from allm.researcher.queue import RecommendationQueue
from allm.researcher.types import KnowledgePackage
from allm.storage.base import RecordStore
from allm.teacher.visual_kel_bridge import (
    TeacherVisualExportResult,
    persist_teacher_packages,
    sync_researcher_packages,
)
from allm.teacher.visual_review_service import TeacherVisualReviewService

NAMESPACE = "teacher_kel_session"
SESSION_KEY = "current"

SessionStatus = Literal["awaiting_review", "exported", "resumed", "complete"]


class TeacherKelSession(BaseModel):
    """Pause point between Researcher distill and KEL visual delivery."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    run_dir: str
    status: SessionStatus = "awaiting_review"
    pending_briefs: int = 0
    total_briefs: int = 0
    student_exports: int = 0
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exported_at: datetime | None = None
    resumed_at: datetime | None = None


class TeacherKelSessionStore:
    """Persist one active Teacher pause session per record store."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    def open(self, run_dir: Path | str, *, pending_briefs: int, total_briefs: int) -> TeacherKelSession:
        """Create a new pause session after Researcher distill."""
        session = TeacherKelSession(
            session_id=f"tsess_{uuid.uuid4().hex[:12]}",
            run_dir=str(run_dir),
            status="awaiting_review",
            pending_briefs=pending_briefs,
            total_briefs=total_briefs,
        )
        self._put(session, reason="teacher pause opened")
        return session

    def get(self) -> TeacherKelSession | None:
        """Return the active session, if any."""
        record = self._store.get(NAMESPACE, SESSION_KEY)
        if record is None:
            return None
        return TeacherKelSession.model_validate(record.value)

    def update(self, session: TeacherKelSession, *, reason: str) -> TeacherKelSession:
        """Persist an updated session snapshot."""
        self._put(session, reason=reason)
        return session

    def mark_exported(self, *, student_exports: int) -> TeacherKelSession | None:
        """Mark session exported after Teacher approves visual packages."""
        session = self.get()
        if session is None:
            return None
        updated = session.model_copy(
            update={
                "status": "exported",
                "student_exports": student_exports,
                "exported_at": datetime.now(timezone.utc),
                "pending_briefs": 0,
            }
        )
        return self.update(updated, reason="teacher visual export")

    def mark_resumed(self) -> TeacherKelSession | None:
        """Mark session resumed before KEL loop continues."""
        session = self.get()
        if session is None:
            return None
        updated = session.model_copy(
            update={
                "status": "resumed",
                "resumed_at": datetime.now(timezone.utc),
            }
        )
        return self.update(updated, reason="kel loop resumed")

    def mark_complete(self) -> TeacherKelSession | None:
        """Mark session complete after KEL loop finishes."""
        session = self.get()
        if session is None:
            return None
        updated = session.model_copy(update={"status": "complete"})
        return self.update(updated, reason="kel loop complete")

    def clear(self) -> None:
        """Remove the active session."""
        if SESSION_KEY in self._store.keys(NAMESPACE):
            self._store.delete(NAMESPACE, SESSION_KEY, reason="session cleared")

    def _put(self, session: TeacherKelSession, *, reason: str) -> None:
        self._store.put(
            NAMESPACE,
            SESSION_KEY,
            json.loads(session.model_dump_json()),
            reason=reason,
        )


def resume_file_path(run_dir: Path | str, override: Path | str | None = None) -> Path:
    """Resolve the resume flag path for one paused run."""
    if override is not None:
        return Path(override)
    return Path(run_dir) / "teacher_resume.flag"


def wait_for_teacher_export(
    store: RecordStore,
    *,
    resume_file: Path | str | None = None,
    timeout_sec: float | None = None,
    poll_sec: float = 2.0,
) -> TeacherKelSession:
    """Block until Teacher exports visuals or a resume flag is touched."""
    session_store = TeacherKelSessionStore(store)
    session = session_store.get()
    if session is None:
        raise RuntimeError("no active Teacher KEL session")

    flag = resume_file_path(session.run_dir, override=resume_file)
    deadline = time.monotonic() + timeout_sec if timeout_sec and timeout_sec > 0 else None

    while True:
        session = session_store.get()
        if session is None:
            raise RuntimeError("Teacher KEL session disappeared")
        if session.status in {"exported", "resumed"}:
            return session
        if flag.is_file():
            return session
        review = TeacherVisualReviewService(store, packages=RecommendationQueue(store).packages())
        summary = review.summary()
        if summary["student_exports"] > 0:
            session_store.mark_exported(student_exports=summary["student_exports"])
            return session_store.get() or session
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Teacher review timed out after {timeout_sec}s")
        time.sleep(poll_sec)


def finalize_teacher_export(
    store: RecordStore,
    researcher: object,
    packages: list[KnowledgePackage] | tuple[KnowledgePackage, ...],
) -> TeacherVisualExportResult:
    """Export approved briefs if needed and sync packages for KEL delivery."""
    queue = RecommendationQueue(store)
    current = queue.packages() or list(packages)
    service = TeacherVisualReviewService(store, packages=current)
    summary = service.summary()

    if summary["student_exports"] == 0:
        exports = service.export_approved()
        updated = service.packages()
        persist_teacher_packages(store, updated)
    else:
        exports = tuple(
            export
            for package in service.packages()
            for export in package.student_visual_packages
        )
        updated = service.packages()

    sync_researcher_packages(researcher, updated)
    final_summary = TeacherVisualReviewService(store, packages=list(updated)).summary()
    return TeacherVisualExportResult(
        packages=updated,
        exports=exports,
        approved_count=final_summary["approved"],
        rejected_count=final_summary["rejected"],
        pending_count=final_summary["pending"],
    )
