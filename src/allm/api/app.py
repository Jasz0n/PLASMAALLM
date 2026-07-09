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

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request

import allm
from allm.api.schemas import (
    ClaimRequest,
    ConceptSummary,
    DocumentSubmission,
    EvidenceSubmission,
    ResolveRequest,
    WebhookApproval,
    WebhookRegistration,
)
from allm.api.security import (
    Principal,
    RateLimiter,
    TokenVerifier,
    default_verifier,
)
from allm.api.dashboard import build_dashboard_router
from allm.api.teacher_visual import build_teacher_visual_router
from allm.events import EventLog, WebhookDispatcher, WebhookRegistry, WebhookSender
from allm.evidence import EvidenceBinder, EvidenceLedger, EvidencePackage
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard, ProposalError
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState


def create_app(
    db_path: Path | str,
    *,
    verifier: TokenVerifier | None = None,
    rate_limiter: RateLimiter | None = None,
    webhook_sender: WebhookSender | None = None,
) -> FastAPI:
    """Build the API over one SQLite-backed record store.

    ``verifier`` is the M50 auth hook point — the platform owns
    identity, the core verifies what it is handed (default: env-driven,
    see :func:`allm.api.security.default_verifier`). Reads stay open;
    every write requires a principal and is rate-limited per principal.
    """
    store = SQLiteRecordStore(db_path)
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    binder = EvidenceBinder(graph, ledger)
    board = ProposalBoard(store, binder)
    documents = DocumentStore(store)
    webhooks = WebhookRegistry(store)
    dispatcher = WebhookDispatcher(webhooks, store, sender=webhook_sender)
    events = EventLog(store, on_emit=dispatcher.dispatch)
    kel = KnowledgeEvaluationLayer(graph, store, KnowledgeState(store), ledger=ledger)

    auth = verifier or default_verifier()
    limiter = rate_limiter or RateLimiter.from_env(os.environ.get("ALLM_API_RATE_LIMIT"))

    def writer(request: Request) -> Principal:
        """Dependency guarding every mutating endpoint."""
        header = request.headers.get("Authorization", "")
        token = header.removeprefix("Bearer ").strip() if header else None
        principal = auth.verify(token)
        if principal is None:
            raise HTTPException(401, "invalid or missing bearer token")
        key = principal.contributor_id if not principal.anonymous else (
            request.client.host if request.client else "unknown"
        )
        if not limiter.allow(key):
            raise HTTPException(429, "rate limit exceeded; slow down")
        return principal

    app = FastAPI(title="ALLM", version=allm.__version__)
    app.include_router(build_teacher_visual_router(store))
    app.include_router(build_dashboard_router(store))

    def to_package(submission: EvidenceSubmission) -> EvidencePackage:
        data = submission.model_dump()
        data["artifacts"] = tuple(submission.artifacts)
        data["reproduction_steps"] = tuple(submission.reproduction_steps)
        data["related_concepts"] = tuple(submission.related_concepts)
        return EvidencePackage.build(**data)

    @app.get("/health")
    def health() -> dict:
        """Liveness: the process is up. Cheap, never touches the store."""
        return {"status": "ok", "version": allm.__version__}

    @app.get("/ready")
    def ready() -> dict:
        """Readiness: the store answers, so it is safe to route traffic here.

        Distinct from ``/health`` on purpose — an orchestrator keeps the
        container alive on ``/health`` but withholds traffic until
        ``/ready`` passes (e.g. the volume mounted and the DB opened).
        """
        try:
            store.get("_readiness", "_probe")  # cheap indexed round-trip
        except Exception as exc:  # pragma: no cover - only on a broken store
            raise HTTPException(503, f"storage not ready: {exc}") from exc
        return {"status": "ready", "version": allm.__version__}

    @app.get("/wire")
    def wire() -> dict:
        """The frozen wire contract (M51) — build against this, not our
        source. Versioned independently of the engine."""
        from allm.wire import wire_contract

        return wire_contract()

    # -- knowledge ------------------------------------------------------

    @app.get("/concepts")
    def list_concepts(
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[ConceptSummary]:
        rows = graph.concepts()[offset : offset + limit]
        return [
            ConceptSummary(
                name=c.name,
                confidence=c.confidence,
                status=c.status,
                evidence_count=len(c.evidence),
            )
            for c in rows
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
    def submit_evidence(
        submission: EvidenceSubmission, principal: Principal = Depends(writer)
    ) -> dict:
        package = to_package(submission)
        breakdown = binder.submit(package)
        events.emit(
            "evidence.submitted",
            package.id,
            {"concept": package.concept, "outcome": package.outcome,
             "contributor": package.contributor},
        )
        events.emit(
            "confidence.changed",
            package.concept,
            {"value": breakdown.value, "contributors": breakdown.contributors,
             "independent_replications": breakdown.independent_replications},
        )
        return {
            "package_id": package.id,
            "confidence": breakdown.model_dump(mode="json"),
        }

    @app.post("/documents", status_code=201)
    def submit_documents(
        submissions: list[DocumentSubmission], principal: Principal = Depends(writer)
    ) -> dict:
        if len(submissions) > 50:
            raise HTTPException(413, "at most 50 documents per request")
        for doc in submissions:
            documents.ingest_text(doc.name, doc.text, context=doc.context)
        result = KDPipeline().distill(documents)
        report = GraphInjector(graph, store).inject(result)
        proposals = []
        for conflict in result.conflicts:
            proposal = board.from_conflict(conflict)
            proposals.append(proposal.id)
            events.emit(
                "proposal.opened",
                proposal.id,
                {"question": proposal.question, "origin": "conflict"},
            )
        return {
            "units": len(result.units),
            "conflicts": len(result.conflicts),
            "graph": report,
            "proposals_opened": proposals,
        }

    # -- proposals -------------------------------------------------------------

    @app.get("/proposals")
    def list_proposals(
        status: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict]:
        rows = board.proposals(status=status)[offset : offset + limit]
        return [p.model_dump(mode="json") for p in rows]

    @app.post("/proposals/{proposal_id}/claim")
    def claim_proposal(
        proposal_id: str, request: ClaimRequest, principal: Principal = Depends(writer)
    ) -> dict:
        try:
            return board.claim(proposal_id, request.contributor).model_dump(mode="json")
        except ProposalError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/proposals/{proposal_id}/resolve")
    def resolve_proposal(
        proposal_id: str, request: ResolveRequest, principal: Principal = Depends(writer)
    ) -> dict:
        try:
            resolved = board.resolve(
                proposal_id, [to_package(p) for p in request.packages]
            )
        except ProposalError as exc:
            raise HTTPException(409, str(exc)) from exc
        events.emit(
            "proposal.resolved",
            resolved.id,
            {"concept": resolved.concept, "status": resolved.status},
        )
        return resolved.model_dump(mode="json")

    # -- operations ------------------------------------------------------

    @app.get("/audit")
    def audit_trail(
        namespace: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict]:
        """Who changed what, when, why — metadata only, values stay
        behind their own endpoints."""
        return [
            {
                "namespace": r.namespace,
                "key": r.key,
                "version": r.version,
                "reason": r.reason,
                "created_at": r.created_at.isoformat(),
            }
            for r in store.audit(namespace, limit=limit, offset=offset)
        ]

    # -- event stream ----------------------------------------------------

    @app.get("/events")
    def event_stream(
        since: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict:
        """The platform's live feed: events with ``seq > since``, in order.

        Poll with ``since`` set to the last ``seq`` you saw; ``cursor`` in
        the response is where to resume so nothing is missed or replayed.
        """
        batch = events.since(since, limit=limit)
        return {
            "events": [e.model_dump(mode="json") for e in batch],
            "cursor": batch[-1].seq if batch else since,
            "total": events.count(),
        }

    # -- webhooks (outbound event delivery, M51) -------------------------

    @app.post("/webhooks", status_code=201)
    def register_webhook(
        registration: WebhookRegistration, principal: Principal = Depends(writer)
    ) -> dict:
        """Register an endpoint for the event feed. It is *proposed* and
        delivers nothing until approved — the secret is returned once."""
        try:
            subscription = webhooks.register(
                registration.url,
                event_types=tuple(registration.event_types),
                secret=registration.secret,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        # the one and only time the secret is exposed
        return subscription.model_dump(mode="json")

    @app.get("/webhooks")
    def list_webhooks() -> list[dict]:
        """Subscriptions with their status — secrets never leave."""
        return [s.public() for s in webhooks.all()]

    @app.post("/webhooks/{subscription_id}/approve")
    def approve_webhook(
        subscription_id: str,
        approval: WebhookApproval,
        principal: Principal = Depends(writer),
    ) -> dict:
        """The named-human approval gate: only now does it start delivering."""
        try:
            approved = webhooks.approve(
                subscription_id, approver=approval.approver, reason=approval.reason
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return approved.public()

    @app.post("/webhooks/{subscription_id}/disable")
    def disable_webhook(
        subscription_id: str, principal: Principal = Depends(writer)
    ) -> dict:
        try:
            disabled = webhooks.disable(subscription_id, reason="disabled via API")
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return disabled.public()

    @app.get("/webhooks/deliveries")
    def webhook_deliveries(
        limit: int = Query(default=50, ge=1, le=500),
    ) -> list[dict]:
        """The outbound audit trail: what was delivered and what failed."""
        return [d.model_dump(mode="json") for d in dispatcher.recent_deliveries(limit=limit)]

    # -- measurement ---------------------------------------------------------------

    @app.post("/kel/evaluate")
    def kel_evaluate(principal: Principal = Depends(writer)) -> dict:
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
