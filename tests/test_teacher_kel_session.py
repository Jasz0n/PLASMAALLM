"""Tests for Teacher KEL pause/resume session (M34)."""

import threading
import time
from pathlib import Path

import pytest

from allm.researcher.multimodal_types import DistilledVisualBrief
from allm.researcher.types import KnowledgePackage
from allm.storage import SQLiteRecordStore
from allm.teacher.teacher_kel_session import (
    TeacherKelSessionStore,
    finalize_teacher_export,
    resume_file_path,
    wait_for_teacher_export,
)
from allm.teacher.visual_export import approve_visual_brief
from allm.teacher.visual_review_service import TeacherVisualReviewService


def _brief(brief_id: str) -> DistilledVisualBrief:
    return DistilledVisualBrief(
        brief_id=brief_id,
        concept_name=f"concept-{brief_id}",
        images=("diagram",),
        evidence_confidence=0.9,
        source_kind="workshop",
    )


def test_session_open_and_mark_exported() -> None:
    store = SQLiteRecordStore(":memory:")
    session_store = TeacherKelSessionStore(store)
    session = session_store.open("/tmp/run", pending_briefs=2, total_briefs=3)
    assert session.status == "awaiting_review"
    exported = session_store.mark_exported(student_exports=1)
    assert exported is not None
    assert exported.status == "exported"
    assert exported.student_exports == 1


def test_wait_for_teacher_export_resume_flag(tmp_path: Path) -> None:
    store = SQLiteRecordStore(":memory:")
    session_store = TeacherKelSessionStore(store)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    session_store.open(run_dir, pending_briefs=1, total_briefs=1)
    flag = resume_file_path(run_dir)

    def touch_flag() -> None:
        time.sleep(0.2)
        flag.touch()

    threading.Thread(target=touch_flag, daemon=True).start()
    result = wait_for_teacher_export(store, resume_file=flag, poll_sec=0.1)
    assert result.session_id


def test_finalize_teacher_export_after_manual_approval() -> None:
    store = SQLiteRecordStore(":memory:")
    package = KnowledgePackage.build(
        provider="kids-workshops",
        title="Workshop",
        curriculum_topic="kids-plasma",
        distilled_visual_briefs=(_brief("dvis_ok"),),
    )
    service = TeacherVisualReviewService(store, packages=[package])
    service.record_approval("dvis_ok", approved=True)

    class _Researcher:
        persisted: list[KnowledgePackage] = []

        def persist_package(self, pkg, *, reason: str = "") -> None:
            self.persisted.append(pkg)

    researcher = _Researcher()
    result = finalize_teacher_export(store, researcher, [package])
    assert len(result.exports) == 1
    assert len(researcher.persisted) == 1
