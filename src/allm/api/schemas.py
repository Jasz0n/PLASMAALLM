"""API request/response schemas.

Kept separate from the domain types on purpose: the wire format is a
public contract the platform depends on, and it must be able to evolve
(or stay frozen) independently of internal models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from allm.evidence.types import Artifact, Outcome, PackageKind


class EvidenceSubmission(BaseModel):
    """POST /evidence body — a package without an id (the core mints it)."""

    claim: str
    concept: str
    contributor: str
    kind: PackageKind = "experiment"
    outcome: Outcome
    artifacts: list[Artifact] = Field(default_factory=list)
    measurements: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    reproduction_steps: list[str] = Field(default_factory=list)
    replicates: str | None = None
    related_concepts: list[str] = Field(default_factory=list)


class DocumentSubmission(BaseModel):
    """POST /documents body — one raw human explanation stream."""

    name: str
    text: str
    context: str = "general"


class ClaimRequest(BaseModel):
    contributor: str


class ResolveRequest(BaseModel):
    packages: list[EvidenceSubmission]


class ConceptSummary(BaseModel):
    name: str
    confidence: float
    status: str
    evidence_count: int


class VisualBriefSummary(BaseModel):
    """Teacher list view for one distilled visual brief."""

    brief_id: str
    concept_name: str
    source_kind: str
    provider: str
    package_id: str
    evidence_confidence: float
    image_count: int
    has_diagram: bool
    approval_status: str
    approved_by: str | None = None
    review_note: str | None = None


class VisualBriefDetail(VisualBriefSummary):
    """Teacher detail view — includes internal notes, not for students."""

    concept_description: str = ""
    images: list[str] = Field(default_factory=list)
    diagram_summary: str | None = None
    explanations: list[str] = Field(default_factory=list)
    experiment_prompt: str | None = None
    questions: list[str] = Field(default_factory=list)
    teacher_notes: str = ""
    source_refs: list[str] = Field(default_factory=list)
    curriculum_topic: str | None = None


class VisualApprovalRequest(BaseModel):
    """POST body for approving or rejecting one brief."""

    approved: bool = True
    max_images: int = Field(default=2, ge=0, le=6)
    max_questions: int = Field(default=3, ge=0, le=10)
    include_diagram: bool = True
    include_experiment: bool = True
    approved_by: str = "teacher-ui"
    review_note: str = ""


class VisualReviewSummary(BaseModel):
    """Dashboard counts for Teacher visual review."""

    total_briefs: int
    pending: int
    approved: int
    rejected: int
    student_exports: int
    workshop_briefs: int
    book_briefs: int


class VisualExportResponse(BaseModel):
    """Result of exporting approved briefs."""

    exports: list[dict[str, Any]]
    export_count: int
    student_exports_total: int


class TeacherKelSessionResponse(BaseModel):
    """Active Teacher pause session between Researcher and KEL loop."""

    session_id: str
    run_dir: str
    status: str
    pending_briefs: int
    total_briefs: int
    student_exports: int
    opened_at: str
    exported_at: str | None = None
    resumed_at: str | None = None
    resume_flag: str
