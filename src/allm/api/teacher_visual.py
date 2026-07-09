"""Teacher visual review HTTP routes and minimal UI (M31)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from allm.api.schemas import (
    TeacherKelSessionResponse,
    VisualApprovalRequest,
    VisualBriefDetail,
    VisualBriefSummary,
    VisualExportResponse,
    VisualReviewSummary,
)
from allm.researcher.queue import RecommendationQueue
from allm.storage.base import RecordStore
from allm.teacher.teacher_kel_session import TeacherKelSessionStore, resume_file_path
from allm.teacher.visual_kel_bridge import persist_teacher_packages
from allm.teacher.visual_review_service import BriefRef, TeacherVisualReviewService

TEACHER_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ALLM Teacher — Visual Review</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 960px; }
    h1 { margin-bottom: 0.25rem; }
    .meta { color: #555; margin-bottom: 1rem; }
    .filters { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .filters button { padding: 0.35rem 0.75rem; cursor: pointer; }
    .filters button.active { background: #1a56db; color: #fff; border-color: #1a56db; }
    .summary { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .summary span { background: #f3f4f6; padding: 0.35rem 0.6rem; border-radius: 4px; }
    .card { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin-bottom: 0.75rem; }
    .card h3 { margin: 0 0 0.35rem; }
    .badge { font-size: 0.75rem; padding: 0.1rem 0.4rem; border-radius: 3px; background: #e5e7eb; }
    .badge.pending { background: #fef3c7; }
    .badge.approved { background: #d1fae5; }
    .badge.rejected { background: #fee2e2; }
    .badge.workshop { background: #dbeafe; }
    .badge.book { background: #ede9fe; }
    .actions { margin-top: 0.75rem; display: flex; gap: 0.5rem; }
    .actions button { cursor: pointer; }
    .detail { font-size: 0.9rem; color: #374151; white-space: pre-wrap; }
    #status { margin-top: 1rem; color: #065f46; }
    .toolbar { margin-bottom: 1rem; }
  </style>
</head>
<body>
  <h1>Teacher Visual Review</h1>
  <p class="meta">Selectively approve workshop and book distilled briefs before student delivery.</p>
  <div class="summary" id="sessionPanel" style="background:#eff6ff;padding:0.75rem;border-radius:6px;margin-bottom:1rem;"></div>
  <div class="summary" id="summary"></div>
  <div class="filters" id="filters"></div>
  <div class="toolbar">
    <button id="exportBtn" type="button">Export approved → student packages</button>
    <button id="resumeBtn" type="button">Signal resume KEL loop</button>
  </div>
  <div id="list"></div>
  <p id="status"></p>
  <script>
    let sourceKind = "";
    let statusFilter = "pending";
    // Works whether served at /teacher or under a proxy prefix like /allm/teacher.
    const PREFIX = location.pathname.replace(/\/teacher\/?$/, "");

    async function loadSession() {
      const res = await fetch(PREFIX + "/teacher/session");
      const panel = document.getElementById("sessionPanel");
      if (!res.ok) {
        panel.innerHTML = "<span>No active KEL pause session.</span>";
        return;
      }
      const data = await res.json();
      panel.innerHTML = [
        `<strong>KEL session ${data.session_id}</strong>`,
        `<span>status: ${data.status}</span>`,
        `<span>pending: ${data.pending_briefs}/${data.total_briefs}</span>`,
        `<span>exports: ${data.student_exports}</span>`,
        `<span>resume flag: ${data.resume_flag}</span>`,
      ].join(" · ");
    }

    async function loadSummary() {
      const res = await fetch(PREFIX + "/teacher/visual-review/summary");
      const data = await res.json();
      document.getElementById("summary").innerHTML = [
        `<span>Total: ${data.total_briefs}</span>`,
        `<span>Pending: ${data.pending}</span>`,
        `<span>Approved: ${data.approved}</span>`,
        `<span>Rejected: ${data.rejected}</span>`,
        `<span>Workshop: ${data.workshop_briefs}</span>`,
        `<span>Book: ${data.book_briefs}</span>`,
        `<span>Student exports: ${data.student_exports}</span>`,
      ].join("");
    }

    function renderFilters() {
      const kinds = [
        ["", "All sources"],
        ["workshop", "Workshop"],
        ["book", "Book"],
      ];
      const statuses = [
        ["pending", "Pending"],
        ["approved", "Approved"],
        ["rejected", "Rejected"],
        ["all", "All"],
      ];
      const kindHtml = kinds.map(([value, label]) =>
        `<button type="button" data-kind="${value}" class="${value === sourceKind ? "active" : ""}">${label}</button>`
      ).join("");
      const statusHtml = statuses.map(([value, label]) =>
        `<button type="button" data-status="${value}" class="${value === statusFilter ? "active" : ""}">${label}</button>`
      ).join("");
      const el = document.getElementById("filters");
      el.innerHTML = kindHtml + statusHtml;
      el.querySelectorAll("[data-kind]").forEach(btn => btn.onclick = () => {
        sourceKind = btn.dataset.kind;
        renderFilters();
        loadBriefs();
      });
      el.querySelectorAll("[data-status]").forEach(btn => btn.onclick = () => {
        statusFilter = btn.dataset.status;
        renderFilters();
        loadBriefs();
      });
    }

    async function decide(briefId, approved) {
      await fetch(PREFIX + `/teacher/visual-briefs/${encodeURIComponent(briefId)}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved, approved_by: "teacher-ui" }),
      });
      document.getElementById("status").textContent = `${approved ? "Approved" : "Rejected"} ${briefId}`;
      await loadSummary();
      await loadBriefs();
    }

    async function loadBriefs() {
      const params = new URLSearchParams({ status: statusFilter });
      if (sourceKind) params.set("source_kind", sourceKind);
      const res = await fetch(PREFIX + `/teacher/visual-briefs?${params}`);
      const rows = await res.json();
      const list = document.getElementById("list");
      if (!rows.length) {
        list.innerHTML = "<p>No briefs match this filter.</p>";
        return;
      }
      list.innerHTML = rows.map(row => `
        <div class="card">
          <h3>${row.concept_name}</h3>
          <span class="badge ${row.approval_status}">${row.approval_status}</span>
          <span class="badge ${row.source_kind || "unknown"}">${row.source_kind || "unknown"}</span>
          <span class="badge">${row.provider}</span>
          <p class="detail">Confidence ${row.evidence_confidence.toFixed(2)} · ${row.image_count} image(s) · ${row.brief_id}</p>
          <div class="actions">
            <button type="button" onclick="decide('${row.brief_id}', true)">Approve</button>
            <button type="button" onclick="decide('${row.brief_id}', false)">Reject</button>
          </div>
        </div>
      `).join("");
    }

    document.getElementById("exportBtn").onclick = async () => {
      const res = await fetch(PREFIX + "/teacher/visual-exports", { method: "POST" });
      const data = await res.json();
      document.getElementById("status").textContent =
        `Exported ${data.export_count} package(s); ${data.student_exports_total} total student exports.`;
      await loadSummary();
      await loadSession();
    };

    document.getElementById("resumeBtn").onclick = async () => {
      const res = await fetch(PREFIX + "/teacher/session/resume", { method: "POST" });
      const data = await res.json();
      document.getElementById("status").textContent =
        `Resume signaled for session ${data.session_id} (${data.status}).`;
      await loadSession();
    };

    renderFilters();
    loadSession();
    loadSummary();
    loadBriefs();
  </script>
</body>
</html>
"""


def _to_summary(service: TeacherVisualReviewService, ref: BriefRef) -> VisualBriefSummary:
    approval = service.approval_record(ref.brief.brief_id)
    status = service.approval_status(ref.brief.brief_id)
    return VisualBriefSummary(
        brief_id=ref.brief.brief_id,
        concept_name=ref.brief.concept_name,
        source_kind=ref.brief.source_kind,
        provider=ref.provider,
        package_id=ref.package_id,
        evidence_confidence=ref.brief.evidence_confidence,
        image_count=len(ref.brief.images),
        has_diagram=ref.brief.diagram_summary is not None,
        approval_status=status,
        approved_by=approval.approved_by if approval else None,
        review_note=approval.review_note if approval else None,
    )


def _to_detail(service: TeacherVisualReviewService, ref: BriefRef) -> VisualBriefDetail:
    base = _to_summary(service, ref)
    return VisualBriefDetail(
        **base.model_dump(),
        concept_description=ref.brief.concept_description,
        images=list(ref.brief.images),
        diagram_summary=ref.brief.diagram_summary,
        explanations=list(ref.brief.explanations),
        experiment_prompt=ref.brief.experiment_prompt,
        questions=list(ref.brief.questions),
        teacher_notes=ref.brief.teacher_notes,
        source_refs=list(ref.brief.source_refs),
        curriculum_topic=ref.curriculum_topic,
    )


def _session_response(session) -> TeacherKelSessionResponse:
    return TeacherKelSessionResponse(
        session_id=session.session_id,
        run_dir=session.run_dir,
        status=session.status,
        pending_briefs=session.pending_briefs,
        total_briefs=session.total_briefs,
        student_exports=session.student_exports,
        opened_at=session.opened_at.isoformat(),
        exported_at=session.exported_at.isoformat() if session.exported_at else None,
        resumed_at=session.resumed_at.isoformat() if session.resumed_at else None,
        resume_flag=str(resume_file_path(session.run_dir)),
    )


def build_teacher_visual_router(
    store: RecordStore,
    *,
    on_packages_updated: Callable[[tuple], None] | None = None,
) -> APIRouter:
    """Create Teacher visual review routes bound to one record store."""
    router = APIRouter(prefix="/teacher", tags=["teacher"])
    queue = RecommendationQueue(store)
    service = TeacherVisualReviewService(store, packages=queue.packages())
    session_store = TeacherKelSessionStore(store)

    def refresh_packages() -> None:
        service.set_packages(queue.packages())

    @router.get("/session", response_model=TeacherKelSessionResponse)
    def get_teacher_session() -> TeacherKelSessionResponse:
        session = session_store.get()
        if session is None:
            raise HTTPException(404, "no active Teacher KEL session")
        return _session_response(session)

    @router.post("/session/resume", response_model=TeacherKelSessionResponse)
    def resume_teacher_session() -> TeacherKelSessionResponse:
        session = session_store.get()
        if session is None:
            raise HTTPException(404, "no active Teacher KEL session")
        flag = resume_file_path(session.run_dir)
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()
        updated = session_store.mark_resumed() or session
        return _session_response(updated)

    @router.get("/", response_class=HTMLResponse)
    def teacher_ui() -> str:
        return TEACHER_UI_HTML

    @router.get("/visual-review/summary", response_model=VisualReviewSummary)
    def review_summary() -> VisualReviewSummary:
        refresh_packages()
        return VisualReviewSummary(**service.summary())

    @router.get("/visual-briefs", response_model=list[VisualBriefSummary])
    def list_visual_briefs(
        source_kind: str | None = None,
        status: Literal["pending", "approved", "rejected", "all"] = "pending",
    ) -> list[VisualBriefSummary]:
        refresh_packages()
        refs = service.list_briefs(source_kind=source_kind, status=status)
        return [_to_summary(service, ref) for ref in refs]

    @router.get("/visual-briefs/{brief_id}", response_model=VisualBriefDetail)
    def get_visual_brief(brief_id: str) -> VisualBriefDetail:
        refresh_packages()
        ref = service.get_brief(brief_id)
        if ref is None:
            raise HTTPException(404, f"unknown visual brief {brief_id!r}")
        return _to_detail(service, ref)

    @router.post("/visual-briefs/{brief_id}/approve", status_code=204)
    def approve_visual_brief_route(brief_id: str, request: VisualApprovalRequest) -> None:
        refresh_packages()
        try:
            service.record_approval(
                brief_id,
                approved=request.approved,
                max_images=request.max_images,
                max_questions=request.max_questions,
                include_diagram=request.include_diagram,
                include_experiment=request.include_experiment,
                approved_by=request.approved_by,
                review_note=request.review_note,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/visual-exports", response_model=VisualExportResponse)
    def export_visual_briefs() -> VisualExportResponse:
        refresh_packages()
        exports = service.export_approved()
        updated = service.packages()
        persist_teacher_packages(store, updated)
        summary = service.summary()
        session_store.mark_exported(student_exports=summary["student_exports"])
        if on_packages_updated is not None:
            on_packages_updated(service.packages())
        return VisualExportResponse(
            exports=[export.model_dump(mode="json") for export in exports],
            export_count=len(exports),
            student_exports_total=summary["student_exports"],
        )

    return router
