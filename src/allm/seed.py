"""``allm seed`` — a reproducible public-loop scenario (Roadmap M52).

Populates a fresh store by *actually running the loop the exit criterion
describes* — discussion → KDP → conflict → proposal → independent
replications → confidence shift — through the real subsystems, not
fabricated rows. So a fresh deploy is non-empty and the dashboard is
alive, and CI rehearses the whole public loop before real contributors
arrive.

The corpus is deliberately Keshe-plasma flavoured (the candidate pilot
domain): two workshop explanations disagree on how long a nano coating
takes to form, and only replicated evidence settles it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.events import EventLog
from allm.evidence import EvidenceBinder, EvidenceLedger, EvidencePackage
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard
from allm.storage.base import RecordStore
from allm.teacher import KnowledgeState

logger = get_logger("seed")

# Discussion: agreeing background + two definitions that disagree on a number.
STARTER_DOCS: tuple[tuple[str, str], ...] = (
    ("workshop-intro-ahmed",
     "Plasma is the fourth state of matter, an ionized gas that conducts electricity."),
    ("workshop-gans-dilek",
     "A gans is a nano-material grown in salt water that holds plasmatic energy in a stable field."),
    ("nano-coating-ahmed",
     "The nano coating is a dark layer that forms after 3 hours in the caustic bath."),
    ("nano-coating-bea",
     "The nano coating is a dark layer that forms after 12 hours in the caustic bath."),
)


class SeedReport(BaseModel):
    """What the rehearsal produced — the loop, made checkable."""

    model_config = ConfigDict(frozen=True)

    concepts: tuple[str, ...]
    contested_concept: str
    proposal_id: str
    proposal_outcome: str
    confidence_before: float
    confidence_after: float
    events: int


def seed_public_loop(store: RecordStore) -> SeedReport:
    """Run the full public loop once and return a checkable report."""
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    binder = EvidenceBinder(graph, ledger)
    board = ProposalBoard(store, binder)
    documents = DocumentStore(store)
    events = EventLog(store)
    kel = KnowledgeEvaluationLayer(graph, store, KnowledgeState(store), ledger=ledger)

    # 1. Discussion → KDP distils concepts (and surfaces the disagreement).
    for name, text in STARTER_DOCS:
        documents.ingest_text(name, text, context="plasma")
    result = KDPipeline().distill(documents)
    GraphInjector(graph, store).inject(result)
    kel.evaluate(distillation=result)  # a baseline measurement for the scorecard

    if not result.conflicts:  # the seed corpus must exercise a real conflict
        raise RuntimeError("seed corpus produced no conflict — the loop needs one")

    # 2. Conflict → proposal (every conflict opens one; announce them).
    proposal = None
    for conflict in result.conflicts:
        opened = board.from_conflict(conflict)
        events.emit(
            "proposal.opened",
            opened.id,
            {"question": opened.question, "concept": opened.concept, "origin": "conflict"},
        )
        proposal = proposal or opened
    assert proposal is not None
    contested = proposal.concept
    before = _confidence_of(graph, contested)

    # 3. A contributor claims the open question.
    board.claim(proposal.id, "ada")

    # 4. Independent replications settle it — the only thing that moves belief.
    first = EvidencePackage.build(
        claim="nano coating fully formed at 12 hours, measured", concept=contested,
        contributor="ada", kind="experiment", outcome="supported", measurements={"hours": 12},
    )
    replications = [
        EvidencePackage.build(
            claim=f"reproduced: coating complete at 12 hours (run {i})", concept=contested,
            contributor=who, kind="replication", outcome="supported",
            replicates=first.id, measurements={"hours": 12},
        )
        for i, who in enumerate(("ben", "cara"), start=1)
    ]
    resolved = board.resolve(proposal.id, [first, *replications])
    breakdown = ledger.confidence(contested)
    after = breakdown.value if breakdown else before

    for package in (first, *replications):
        events.emit(
            "evidence.submitted", package.id,
            {"concept": contested, "outcome": "supported", "contributor": package.contributor},
        )
    events.emit(
        "confidence.changed", contested,
        {"value": after, "contributors": breakdown.contributors,
         "independent_replications": breakdown.independent_replications} if breakdown else {"value": after},
    )
    events.emit(
        "proposal.resolved", resolved.id,
        {"concept": contested, "status": resolved.status,
         "outcome": resolved.resolution.outcome if resolved.resolution else "resolved"},
    )

    # 5. A little more supported evidence elsewhere, so the graph looks lived-in.
    for who, claim, kind in [
        ("ada", "a plasma lamp lights when energized", "experiment"),
        ("dan", "reproduced: the plasma lamp lit again", "replication"),
    ]:
        pkg = EvidencePackage.build(
            claim=claim, concept="Plasma", contributor=who, kind=kind, outcome="supported",
        )
        plasma = binder.submit(pkg)
        events.emit("evidence.submitted", pkg.id,
                    {"concept": "Plasma", "outcome": "supported", "contributor": who})
        events.emit("confidence.changed", "Plasma", {"value": plasma.value})

    kel.evaluate(distillation=result)  # a second measurement, so trends have two points

    concepts = tuple(c.name for c in graph.concepts())
    outcome = resolved.resolution.outcome if resolved.resolution else "resolved"
    logger.info(
        "seed: %d concepts, contested %r resolved %s (%.2f→%.2f), %d events",
        len(concepts), contested, outcome, before, after, events.count(),
    )
    return SeedReport(
        concepts=concepts,
        contested_concept=contested,
        proposal_id=proposal.id,
        proposal_outcome=outcome,
        confidence_before=round(before, 4),
        confidence_after=round(after, 4),
        events=events.count(),
    )


def _confidence_of(graph: KnowledgeGraph, name: str) -> float:
    concept = graph.get(name)
    return concept.confidence if concept else 0.0
