"""The ALLM HTTP API — the boundary the platform talks to.

Design decisions
----------------
- One factory, one SQLite path: ``create_app(db)`` wires the whole core
  (graph, ledger, binder, board, KDP, KEL) over a single record store.
  Identity, incentives and file storage are the platform's job; the
  core receives opaque contributor ids and artifact URIs.
- Submitting documents auto-opens proposals for detected conflicts:
  the vision's "AI suggests what to test next" happens at ingestion,
  not by an operator remembering to ask.
- KEL evaluation is a POST: taking a measurement appends to the time
  series, and mutating reads make trends lie.

Run locally:  uvicorn --factory 'allm.api.app:create_default_app'
Requires the api extras: pip install -e ".[api]"
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

import allm
from allm.api.schemas import (
    ClaimRequest,
    ConceptSummary,
    DocumentSubmission,
    EvidenceSubmission,
    ResolveRequest,
)
from allm.api.teacher_visual import build_teacher_visual_router
from allm.evidence import EvidenceBinder, EvidenceLedger, EvidencePackage
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard, ProposalError
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState


def create_app(db_path: Path | str) -> FastAPI:
    """Build the API over one SQLite-backed record store."""
    store = SQLiteRecordStore(db_path)
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    binder = EvidenceBinder(graph, ledger)
    board = ProposalBoard(store, binder)
    documents = DocumentStore(store)
    kel = KnowledgeEvaluationLayer(graph, store, KnowledgeState(store))

    app = FastAPI(title="ALLM", version=allm.__version__)
    app.include_router(build_teacher_visual_router(store))

    def to_package(submission: EvidenceSubmission) -> EvidencePackage:
        data = submission.model_dump()
        data["artifacts"] = tuple(submission.artifacts)
        data["reproduction_steps"] = tuple(submission.reproduction_steps)
        data["related_concepts"] = tuple(submission.related_concepts)
        return EvidencePackage.build(**data)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": allm.__version__}

    # -- knowledge ------------------------------------------------------

    @app.get("/concepts")
    def list_concepts() -> list[ConceptSummary]:
        return [
            ConceptSummary(
                name=c.name,
                confidence=c.confidence,
                status=c.status,
                evidence_count=len(c.evidence),
            )
            for c in graph.concepts()
        ]

    @app.get("/concepts/{name}")
    def get_concept(name: str) -> dict:
        concept = graph.get(name)
        if concept is None:
            raise HTTPException(404, f"unknown concept {name!r}")
        breakdown = ledger.confidence(name)
        return {
            "concept": concept.model_dump(mode="json"),
            "versions": len(graph.history(name)),
            "provenance": binder.why(name),
            "evidential_confidence": None
            if breakdown is None
            else breakdown.model_dump(mode="json"),
        }

    # -- contributions ------------------------------------------------------

    @app.post("/evidence", status_code=201)
    def submit_evidence(submission: EvidenceSubmission) -> dict:
        package = to_package(submission)
        breakdown = binder.submit(package)
        return {
            "package_id": package.id,
            "confidence": breakdown.model_dump(mode="json"),
        }

    @app.post("/documents", status_code=201)
    def submit_documents(submissions: list[DocumentSubmission]) -> dict:
        for doc in submissions:
            documents.ingest_text(doc.name, doc.text, context=doc.context)
        result = KDPipeline().distill(documents)
        report = GraphInjector(graph, store).inject(result)
        proposals = [board.from_conflict(c).id for c in result.conflicts]
        return {
            "units": len(result.units),
            "conflicts": len(result.conflicts),
            "graph": report,
            "proposals_opened": proposals,
        }

    # -- proposals -------------------------------------------------------------

    @app.get("/proposals")
    def list_proposals(status: str | None = None) -> list[dict]:
        return [p.model_dump(mode="json") for p in board.proposals(status=status)]

    @app.post("/proposals/{proposal_id}/claim")
    def claim_proposal(proposal_id: str, request: ClaimRequest) -> dict:
        try:
            return board.claim(proposal_id, request.contributor).model_dump(mode="json")
        except ProposalError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/proposals/{proposal_id}/resolve")
    def resolve_proposal(proposal_id: str, request: ResolveRequest) -> dict:
        try:
            resolved = board.resolve(
                proposal_id, [to_package(p) for p in request.packages]
            )
        except ProposalError as exc:
            raise HTTPException(409, str(exc)) from exc
        return resolved.model_dump(mode="json")

    # -- measurement ---------------------------------------------------------------

    @app.post("/kel/evaluate")
    def kel_evaluate() -> dict:
        report = kel.evaluate()
        return {
            "report": report.model_dump(mode="json"),
            "findings": [f.model_dump() for f in kel.diagnose()],
        }

    return app


def create_default_app() -> FastAPI:
    """Factory for ``uvicorn --factory``; honours ALLM_STORAGE__PATH."""
    from allm.core.config import load_config

    config = load_config().resolved()
    config.storage.path.parent.mkdir(parents=True, exist_ok=True)
    return create_app(config.storage.path)
