"""API request/response schemas.

Kept separate from the domain types on purpose: the wire format is a
public contract the platform depends on, and it must be able to evolve
(or stay frozen) independently of internal models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from allm.evidence.types import Artifact, Outcome, PackageKind


class EvidenceSubmission(BaseModel):
    """POST /evidence body — a package without an id (the core mints it).

    Size caps (M50): generous for honest contributions, hostile to
    payload abuse. Artifacts are references, never blobs — the platform
    stores files.
    """

    claim: str = Field(min_length=1, max_length=2_000)
    concept: str = Field(min_length=1, max_length=200)
    contributor: str = Field(min_length=1, max_length=200)
    kind: PackageKind = "experiment"
    outcome: Outcome
    artifacts: list[Artifact] = Field(default_factory=list, max_length=32)
    measurements: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    reproduction_steps: list[str] = Field(default_factory=list, max_length=64)
    replicates: str | None = Field(default=None, max_length=200)
    related_concepts: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("measurements", "environment")
    @classmethod
    def _cap_mapping(cls, value: dict) -> dict:
        if len(value) > 128:
            raise ValueError("at most 128 entries")
        return value

    @field_validator("reproduction_steps", "related_concepts")
    @classmethod
    def _cap_items(cls, value: list[str]) -> list[str]:
        for item in value:
            if len(item) > 2_000:
                raise ValueError("entries are capped at 2000 characters")
        return value


class DocumentSubmission(BaseModel):
    """POST /documents body — one raw human explanation stream."""

    name: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=500_000)
    context: str = Field(default="general", max_length=100)


class ClaimRequest(BaseModel):
    contributor: str = Field(min_length=1, max_length=200)


class ResolveRequest(BaseModel):
    packages: list[EvidenceSubmission] = Field(max_length=16)


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
