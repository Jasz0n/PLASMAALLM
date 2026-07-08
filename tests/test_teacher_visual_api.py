"""Tests for Teacher visual review HTTP API (M31)."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from allm.api.app import create_app  # noqa: E402
from allm.researcher.multimodal_types import DistilledVisualBrief  # noqa: E402
from allm.researcher.queue import RecommendationQueue  # noqa: E402
from allm.researcher.types import KnowledgePackage  # noqa: E402


def _seed_packages(store) -> None:
    workshop_brief = DistilledVisualBrief(
        brief_id="dvis_workshop",
        concept_name="plasma motion",
        concept_description="Workshop magnet demo",
        images=("Rotating magnets",),
        evidence_confidence=0.9,
        source_kind="workshop",
        teacher_notes="Teacher only",
    )
    book_brief = DistilledVisualBrief(
        brief_id="dvis_book",
        concept_name="universal order",
        concept_description="Book figure excerpt",
        images=("Diagram of fields",),
        evidence_confidence=0.8,
        source_kind="book",
        teacher_notes="Book internal note",
    )
    queue = RecommendationQueue(store)
    queue.store_package(
        KnowledgePackage.build(
            provider="kids-workshops",
            title="Workshop visuals",
            curriculum_topic="kids-plasma",
            distilled_visual_briefs=(workshop_brief,),
        )
    )
    queue.store_package(
        KnowledgePackage.build(
            provider="keshe-books",
            title="Book visuals",
            curriculum_topic="kids-plasma",
            distilled_visual_briefs=(book_brief,),
        )
    )


@pytest.fixture()
def client(tmp_path: Path):
    store_path = tmp_path / "teacher-ui.sqlite3"
    from allm.storage import SQLiteRecordStore

    store = SQLiteRecordStore(store_path)
    _seed_packages(store)
    store.close()
    app = create_app(store_path)
    with TestClient(app) as test_client:
        yield test_client


def test_teacher_ui_page(client: TestClient) -> None:
    response = client.get("/teacher/")
    assert response.status_code == 200
    assert "Teacher Visual Review" in response.text


def test_list_and_approve_briefs(client: TestClient) -> None:
    summary = client.get("/teacher/visual-review/summary").json()
    assert summary["total_briefs"] == 2
    assert summary["pending"] == 2
    assert summary["workshop_briefs"] == 1
    assert summary["book_briefs"] == 1

    book_rows = client.get("/teacher/visual-briefs", params={"source_kind": "book"}).json()
    assert len(book_rows) == 1
    assert book_rows[0]["brief_id"] == "dvis_book"

    detail = client.get("/teacher/visual-briefs/dvis_workshop").json()
    assert detail["teacher_notes"] == "Teacher only"
    assert detail["source_kind"] == "workshop"

    approve = client.post(
        "/teacher/visual-briefs/dvis_workshop/approve",
        json={"approved": True, "approved_by": "teacher-test"},
    )
    assert approve.status_code == 204

    reject = client.post(
        "/teacher/visual-briefs/dvis_book/approve",
        json={"approved": False, "review_note": "Needs revision"},
    )
    assert reject.status_code == 204

    approved_rows = client.get(
        "/teacher/visual-briefs",
        params={"status": "approved"},
    ).json()
    assert len(approved_rows) == 1
    assert approved_rows[0]["brief_id"] == "dvis_workshop"


def test_export_approved_creates_student_packages(client: TestClient) -> None:
    client.post(
        "/teacher/visual-briefs/dvis_workshop/approve",
        json={"approved": True},
    )
    client.post(
        "/teacher/visual-briefs/dvis_book/approve",
        json={"approved": False},
    )

    export = client.post("/teacher/visual-exports").json()
    assert export["export_count"] == 1
    assert export["student_exports_total"] == 1
    assert export["exports"][0]["concept_name"] == "plasma motion"

    summary = client.get("/teacher/visual-review/summary").json()
    assert summary["student_exports"] == 1


def test_teacher_session_lifecycle(tmp_path: Path) -> None:
    store_path = tmp_path / "session.sqlite3"
    from allm.storage import SQLiteRecordStore
    from allm.teacher.teacher_kel_session import TeacherKelSessionStore

    store = SQLiteRecordStore(store_path)
    _seed_packages(store)
    TeacherKelSessionStore(store).open(tmp_path / "run", pending_briefs=2, total_briefs=2)
    store.close()

    app = create_app(store_path)
    with TestClient(app) as client:
        session = client.get("/teacher/session").json()
        assert session["status"] == "awaiting_review"
        assert session["pending_briefs"] == 2
        assert "teacher_resume.flag" in session["resume_flag"]

        client.post(
            "/teacher/visual-briefs/dvis_workshop/approve",
            json={"approved": True},
        )
        export = client.post("/teacher/visual-exports").json()
        assert export["export_count"] == 1

        resumed = client.post("/teacher/session/resume").json()
        assert resumed["status"] == "resumed"
        assert Path(resumed["resume_flag"]).is_file()
