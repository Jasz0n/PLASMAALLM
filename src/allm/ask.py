"""Ask ALLM — grounded question answering (Roadmap M52).

The anti-oracle chat: every answer is built *only* from the evidence
graph and always carries its confidence and provenance. It never invents
belief. When nothing in the knowledge base matches, it says so and points
to how you'd contribute — which is itself the thesis in action
(*documents propose, evidence disposes*).

Deterministic and offline by default: retrieval is token overlap over
concept names, descriptions and evidence claims, and the answer is
composed from the concept's own recorded state. A natural-language
phrasing layer over a local model can wrap this later, but the grounding
— confidence, provenance, contested-ness — is computed here, not guessed.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from allm.evidence.ledger import EvidenceBinder, EvidenceLedger
from allm.knowledge.graph import KnowledgeGraph
from allm.proposals.board import ProposalBoard

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the is are was were be been being of to in on at for and or but with "
    "how what when where why which who does do did can could would should will "
    "it its this that these those as by from about into more most some any then "
    "much long many take takes form forms".split()
)

# Established belief lives at/above KEL's evidence threshold; below it, a
# claim is still earning its confidence (KELConfig.evidence_confidence_cap).
ESTABLISHED = 0.75


class GroundedAnswer(BaseModel):
    """An answer that can be traced to its evidence — or an honest 'no'."""

    model_config = ConfigDict(frozen=True)

    query: str
    found: bool
    status: str  # established | emerging | contested | unfounded | unknown
    answer: str
    concept: str | None = None
    confidence: float | None = None
    contributors: int = 0
    independent_replications: int = 0
    provenance: str | None = None
    sources: tuple[str, ...] = ()  # evidence package ids behind the number
    related: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()  # unresolved proposals on this concept
    suggestion: str | None = None  # how to contribute, when evidence is thin


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1}


def answer_question(
    query: str,
    graph: KnowledgeGraph,
    ledger: EvidenceLedger,
    board: ProposalBoard,
    *,
    binder: EvidenceBinder | None = None,
) -> GroundedAnswer:
    """Answer ``query`` strictly from the evidence graph."""
    q_tokens = _content_tokens(query)
    concept = _best_match(query, q_tokens, graph, ledger) if q_tokens else None

    if concept is None:
        return GroundedAnswer(
            query=query,
            found=False,
            status="unknown",
            answer=(
                "I don't have evidence about that yet — nothing in the knowledge "
                "base matches your question. I won't guess: a claim earns "
                "confidence here only through submitted, replicated evidence."
            ),
            suggestion=(
                "Contribute what you know as a document or an evidence package; "
                "it will start as a proposal and gain confidence through replication."
            ),
        )

    breakdown = ledger.confidence(concept.name)
    packages = ledger.packages_for(concept.name)
    value = breakdown.value if breakdown else concept.confidence
    proposals = [p for p in board.proposals() if p.concept == concept.name]
    open_questions = tuple(
        p.question for p in proposals if p.status in ("open", "claimed")
    )
    provenance = binder.why(concept.name) if binder is not None else None

    status, answer, suggestion = _compose(
        concept, value, breakdown, packages, open_questions
    )
    return GroundedAnswer(
        query=query,
        found=True,
        status=status,
        answer=answer,
        concept=concept.name,
        confidence=round(value, 4),
        contributors=breakdown.contributors if breakdown else 0,
        independent_replications=breakdown.independent_replications if breakdown else 0,
        provenance=provenance,
        sources=breakdown.packages if breakdown else (),
        related=concept.related,
        open_questions=open_questions,
        suggestion=suggestion,
    )


def _best_match(query: str, q_tokens: set[str], graph: KnowledgeGraph, ledger: EvidenceLedger):
    """Highest-overlap active concept, or None if nothing genuinely matches."""
    best = None
    best_score = 0
    for concept in graph.concepts():
        if concept.status != "active":
            continue
        name_tokens = _content_tokens(concept.name)
        body = concept.description + " " + " ".join(concept.related)
        for package in ledger.packages_for(concept.name):
            body += " " + package.claim
        body_tokens = _content_tokens(body)
        score = 2 * len(q_tokens & name_tokens) + len(q_tokens & body_tokens)
        if score > best_score:
            best_score, best = score, concept
    return best  # score 0 -> None (no shared content word = no honest match)


def _compose(concept, value, breakdown, packages, open_questions):
    """The grounded sentence + status + optional contribution nudge."""
    what = concept.description.strip() or f"{concept.name}"
    conf = f"confidence {value:.2f}"
    who = (
        f"{breakdown.contributors} contributor(s), "
        f"{breakdown.independent_replications} independent replication(s)"
        if breakdown else "no submitted evidence"
    )

    if open_questions:
        return (
            "contested",
            f"{concept.name} is still contested — {conf}, and nothing is settled "
            f"until replications resolve the open question. What is recorded: {what}.",
            None,
        )
    if not packages:
        return (
            "unfounded",
            f"Documents describe {concept.name} ({what}), but no evidence has been "
            f"submitted to back it. A document only proposes a claim — evidence "
            f"disposes — so treat this as unverified ({conf}).",
            "Submit an evidence package to put this on firmer ground.",
        )
    if value >= ESTABLISHED:
        return (
            "established",
            f"{what}. The evidence supports this — {conf} from {who}.",
            None,
        )
    return (
        "emerging",
        f"{what}. This is supported but not yet well-founded — {conf} from {who}; "
        f"more independent replication would strengthen it.",
        "An independent replication would raise its confidence.",
    )
