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
    intent: str  # how_to | quantity | definition
    status: str  # established | emerging | contested | unfounded | procedure | no_procedure | unknown
    engine: str = "extractive"  # extractive | model (grounded RAG)
    answer: str
    concept: str | None = None
    confidence: float | None = None
    contributors: int = 0
    independent_replications: int = 0
    steps: tuple[str, ...] = ()  # the reproducible procedure, for how-to questions
    provenance: str | None = None
    sources: tuple[str, ...] = ()  # evidence package ids behind the number
    related: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()  # unresolved proposals on this concept
    suggestion: str | None = None  # how to contribute, when evidence is thin


_QUANTITY = ("how long", "how much", "how many", "how hot", "how fast", "what temperature")
_HOW_TO = ("how do i", "how to", "how can i", "how would i", "how is it made",
           " make ", " build ", " create ", " prepare ", " produce ", " grow ",
           "steps", "recipe", "procedure", "instructions")


def _intent(query: str) -> str:
    """What kind of answer the question is asking for."""
    q = f" {query.lower().strip()} "
    if any(p in q for p in _QUANTITY):
        return "quantity"
    if any(p in q for p in _HOW_TO) or q.lstrip().startswith(("make ", "build ", "create ")):
        return "how_to"
    return "definition"


def _procedure_steps(packages) -> tuple[str, ...]:
    """The richest reproducible procedure across a concept's evidence."""
    best: tuple[str, ...] = ()
    for package in packages:
        if len(package.reproduction_steps) > len(best):
            best = tuple(package.reproduction_steps)
    return best


def _evidence_status(value, breakdown, packages, open_questions) -> str:
    if open_questions:
        return "contested"
    if not packages:
        return "unfounded"
    if value >= ESTABLISHED:
        return "established"
    return "emerging"


# -- grounded RAG (a local model understands the question; the graph is the
#    only source of fact) ----------------------------------------------------

GROUNDED_PROMPT = (
    "You are ALLM's assistant for an evidence-based knowledge system. Answer the "
    "user's question using ONLY the evidence below.\n"
    "Rules:\n"
    "- Use only the evidence provided. Never add facts from your own knowledge.\n"
    "- If the evidence does not answer the question, say clearly that there is no "
    "evidence for it yet, and do not guess.\n"
    "- Answer exactly what is asked: methods/steps -> list the recorded ones; "
    "'what is' -> define it; a quantity -> give it; 'what exists' -> enumerate "
    "what is recorded and say if only one (or none) is.\n"
    "- State the confidence, and flag when a claim is contested or weakly supported.\n"
    "- Be concise and factual. Never invent steps, numbers, methods or sources.\n\n"
    "EVIDENCE\n{context}\n\n"
    "QUESTION: {query}\n"
    "ANSWER:"
)


def _ranked_concepts(query, q_tokens, graph, ledger, top_n):
    scored = []
    for concept in graph.concepts():
        if concept.status != "active":
            continue
        name_tokens = _content_tokens(concept.name)
        body = concept.description + " " + " ".join(concept.related)
        for package in ledger.packages_for(concept.name):
            body += " " + package.claim + " " + " ".join(package.reproduction_steps)
        score = 2 * len(q_tokens & name_tokens) + len(q_tokens & _content_tokens(body))
        if score > 0:
            scored.append((score, concept))
    scored.sort(key=lambda t: (-t[0], t[1].name))
    return [concept for _, concept in scored[:top_n]]


def _concept_context(concept, ledger, board) -> str:
    """A compact, factual dossier on one concept for the model to ground on."""
    lines = [f"Concept: {concept.name}"]
    lines.append(f"  definition: {concept.description.strip() or '(none recorded)'}")
    breakdown = ledger.confidence(concept.name)
    packages = ledger.packages_for(concept.name)
    if breakdown:
        lines.append(
            f"  confidence: {breakdown.value:.2f} from {breakdown.contributors} "
            f"contributor(s), {breakdown.independent_replications} independent replication(s)"
        )
    else:
        lines.append(f"  confidence: {concept.confidence:.2f} (no submitted evidence packages)")
    for package in packages[:5]:
        lines.append(f"  evidence [{package.kind}/{package.outcome}] by {package.contributor}: {package.claim}")
        if package.reproduction_steps:
            lines.append("    recorded procedure/steps: " + " | ".join(package.reproduction_steps))
    for proposal in [p for p in board.proposals() if p.concept == concept.name]:
        if proposal.status in ("open", "claimed"):
            lines.append(f"  CONTESTED — open question: {proposal.question}")
        elif proposal.resolution:
            lines.append(f"  resolved {proposal.resolution.outcome}: {proposal.question}")
    if concept.related:
        lines.append(f"  related concepts: {', '.join(concept.related)}")
    return "\n".join(lines)


def answer_with_model(query, graph, ledger, board, model, *, binder=None, top_n=3):
    """Grounded RAG: the model composes the answer, but only from the retrieved
    evidence. Facts and provenance are still computed here, not by the model."""
    from allm.models import GenerationParams

    q_tokens = _content_tokens(query)
    ranked = _ranked_concepts(query, q_tokens, graph, ledger, top_n) if q_tokens else []
    if not ranked:
        return GroundedAnswer(
            query=query, found=False, intent=_intent(query), status="unknown", engine="model",
            answer=(
                "I don't have evidence about that yet — nothing in the knowledge base "
                "matches your question. I won't guess."
            ),
            suggestion=(
                "Contribute what you know as a document or an evidence package; it earns "
                "confidence only through replication."
            ),
        )

    context = "\n\n".join(_concept_context(c, ledger, board) for c in ranked)
    prompt = GROUNDED_PROMPT.format(context=context, query=query)
    text = model.generate(
        prompt, GenerationParams(temperature=0.2, max_new_tokens=400)
    ).strip()

    primary = ranked[0]
    breakdown = ledger.confidence(primary.name)
    packages = ledger.packages_for(primary.name)
    value = breakdown.value if breakdown else primary.confidence
    open_questions = tuple(
        p.question for p in board.proposals()
        if p.concept == primary.name and p.status in ("open", "claimed")
    )
    return GroundedAnswer(
        query=query, found=True, intent=_intent(query),
        status=_evidence_status(value, breakdown, packages, open_questions), engine="model",
        answer=text, concept=primary.name, confidence=round(value, 4),
        contributors=breakdown.contributors if breakdown else 0,
        independent_replications=breakdown.independent_replications if breakdown else 0,
        provenance=binder.why(primary.name) if binder is not None else None,
        sources=breakdown.packages if breakdown else (),
        related=primary.related, open_questions=open_questions,
    )


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
    intent = _intent(query)
    q_tokens = _content_tokens(query)
    concept = _best_match(query, q_tokens, graph, ledger) if q_tokens else None

    if concept is None:
        return GroundedAnswer(
            query=query,
            found=False,
            intent=intent,
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
    steps = _procedure_steps(packages) if intent == "how_to" else ()

    status, answer, suggestion = _compose(
        concept, intent, value, breakdown, packages, open_questions, steps
    )
    return GroundedAnswer(
        query=query,
        found=True,
        intent=intent,
        status=status,
        answer=answer,
        concept=concept.name,
        confidence=round(value, 4),
        contributors=breakdown.contributors if breakdown else 0,
        independent_replications=breakdown.independent_replications if breakdown else 0,
        steps=steps,
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


def _compose(concept, intent, value, breakdown, packages, open_questions, steps):
    """The grounded sentence + status + optional contribution nudge."""
    what = concept.description.strip() or f"{concept.name}"
    name = concept.name.lower()
    conf = f"confidence {value:.2f}"
    who = (
        f"{breakdown.contributors} contributor(s), "
        f"{breakdown.independent_replications} independent replication(s)"
        if breakdown else "no submitted evidence"
    )

    # A how-to is answered by the reproducible procedure behind the evidence —
    # not by the definition. If no steps were submitted, say so honestly.
    if intent == "how_to":
        if steps:
            return (
                "procedure",
                f"Here's how to make {name} — a procedure reproduced by {who} ({conf}):",
                None,
            )
        return (
            "no_procedure",
            f"No reproducible procedure for {name} has been submitted — I only have "
            f"what it *is*: {what}. I won't invent the steps. Contribute them as an "
            f"evidence package's reproduction steps and they'll earn confidence "
            f"through replication.",
            "Record the how-to as an evidence package with reproduction steps.",
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
