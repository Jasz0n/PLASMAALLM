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

import asyncio
import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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
from allm.api.chat import build_chat_router
from allm.api.dashboard import build_dashboard_router
from allm.api.teacher_visual import build_teacher_visual_router
from allm.events import EventLog, WebhookDispatcher, WebhookRegistry, WebhookSender
from allm.evidence import EvidenceBinder, EvidenceLedger, EvidencePackage
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard, ProposalError
from allm.core.logging import get_logger
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState

logger = get_logger("api")


# SSE tail poll interval — the store is poll-based, so a live stream checks
# for new events this often when idle. Only affects ``live=true`` streams.
_SSE_POLL_SECONDS = float(os.environ.get("ALLM_SSE_POLL_SECONDS", "1.0"))


def _cors_from_env() -> list[str]:
    """Allowed browser origins from ``ALLM_API_CORS_ORIGINS`` (comma-sep).

    Empty by default — same-origin only, so a fresh deploy is not
    accidentally reachable from any site. ``*`` allows all (dev only).
    """
    raw = os.environ.get("ALLM_API_CORS_ORIGINS", "").strip()
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(
    db_path: Path | str,
    *,
    verifier: TokenVerifier | None = None,
    rate_limiter: RateLimiter | None = None,
    webhook_sender: WebhookSender | None = None,
    cors_origins: list[str] | None = None,
    root_path: str = "",
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

    # Grounded-RAG chat: a local model composes /ask answers from retrieved
    # evidence when ALLM_ASK_MODEL is set (loaded once, reachability permitting).
    ask_model_state: dict = {"loaded": False, "model": None}

    def _ask_model():
        if not ask_model_state["loaded"]:
            ask_model_state["loaded"] = True
            model_id = os.environ.get("ALLM_ASK_MODEL", "").strip()
            if model_id:
                try:
                    from allm.models import ModelSpec, load_model

                    spec = ModelSpec(
                        name="ask",
                        provider=os.environ.get("ALLM_ASK_MODEL_PROVIDER", "ollama"),
                        model_id=model_id,
                        base_url=os.environ.get("OLLAMA_BASE_URL") or None,
                    )
                    ask_model_state["model"] = load_model(spec)
                    logger.info("Ask ALLM using grounded model %s", model_id)
                except Exception as exc:  # keep the API up; fall back to extractive
                    logger.warning("ask model %r unavailable, staying extractive: %s", model_id, exc)
        return ask_model_state["model"]

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

    # root_path tells FastAPI the external prefix a proxy serves us under
    # (e.g. "/allm"), so the interactive /docs and OpenAPI URLs are correct
    # behind a prefix-stripping reverse proxy. The routes themselves are
    # unchanged; the browser UIs derive their own prefix client-side.
    app = FastAPI(title="ALLM", version=allm.__version__, root_path=root_path)

    # -- browser-ready boundary (M52) -----------------------------------
    origins = cors_origins if cors_origins is not None else _cors_from_env()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "Last-Event-ID"],
            # bearer tokens ride the Authorization header, not cookies, so
            # credentialed CORS is unnecessary — and keeping it off lets a
            # wildcard origin stay valid for open reads.
            allow_credentials=False,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """One predictable error shape for clients (M52).

        Keeps FastAPI's ``detail`` for anyone already reading it, and adds
        a structured ``error`` object a frontend can switch on.
        """
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {"status": exc.status_code, "message": exc.detail, "type": "http_error"},
                "detail": exc.detail,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = jsonable_encoder(exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "status": 422,
                    "message": "request validation failed",
                    "type": "validation_error",
                    "fields": fields,
                },
                "detail": fields,
            },
        )

    app.include_router(build_teacher_visual_router(store))
    app.include_router(build_dashboard_router(store))
    app.include_router(build_chat_router())

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

    @app.get("/ask")
    def ask(q: str = Query(..., min_length=1, max_length=500)) -> dict:
        """Grounded Q&A: an answer built only from the evidence graph, with
        its confidence and provenance — or an honest 'no evidence yet'. It
        never invents belief (M52).

        With ``ALLM_ASK_MODEL`` set, a local model *understands* the question
        and composes the answer from the retrieved evidence (grounded RAG);
        otherwise it falls back to deterministic extraction. Either way the
        facts and provenance come from the graph, not the model."""
        from allm.ask import answer_question, answer_with_model

        model = _ask_model()
        if model is not None:
            try:
                return answer_with_model(
                    q, graph, ledger, board, model, binder=binder
                ).model_dump(mode="json")
            except Exception as exc:  # model down/slow → never fail the request
                logger.warning("ask model failed, using extractive fallback: %s", exc)
        return answer_question(q, graph, ledger, board, binder=binder).model_dump(mode="json")

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

    @app.get("/events/stream")
    async def event_sse(
        request: Request,
        since: int = Query(default=0, ge=0),
        live: bool = Query(default=True),
    ) -> StreamingResponse:
        """The live feed as Server-Sent Events — for a browser client.

        Resumes from the ``Last-Event-ID`` header (the browser sends it on
        reconnect) or ``?since=<seq>``, over the same monotonic ``seq``,
        so nothing is missed or replayed. ``live=false`` streams the
        current backlog and closes (handy for a simple catch-up client).
        """
        header_id = request.headers.get("Last-Event-ID")
        cursor = int(header_id) if header_id and header_id.isdigit() else since

        async def generate():
            nonlocal cursor
            yield "retry: 3000\n\n"  # tell the browser how fast to reconnect
            while True:
                batch = events.since(cursor, limit=200)
                if batch:
                    for event in batch:
                        cursor = event.seq
                        data = json.dumps(event.model_dump(mode="json"))
                        yield f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"
                    continue
                if not live or await request.is_disconnected():
                    break
                yield ": keepalive\n\n"  # comment frame keeps idle proxies open
                await asyncio.sleep(_SSE_POLL_SECONDS)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
    """Factory for ``uvicorn --factory``; honours ALLM_STORAGE__PATH and,
    for subpath hosting behind a proxy, ALLM_ROOT_PATH (e.g. ``/allm``)."""
    from allm.core.config import load_config

    config = load_config().resolved()
    config.storage.path.parent.mkdir(parents=True, exist_ok=True)
    return create_app(
        config.storage.path,
        root_path=os.environ.get("ALLM_ROOT_PATH", "").rstrip("/"),
    )
