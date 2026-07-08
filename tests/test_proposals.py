"""Tests for allm.proposals: lifecycle, evidence-driven resolution, factories."""

from pathlib import Path

import pytest

from allm.debate import DebateEngine
from allm.evidence import EvidenceBinder, EvidenceLedger, EvidencePackage
from allm.exam import Question
from allm.kdp.types import ConflictNode, SpanRef
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard, ProposalError
from allm.storage import SQLiteRecordStore
from allm.students import ScriptedStudent

CONCEPT = "Energy Converter"


@pytest.fixture()
def env(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "proposals.sqlite3")
    graph = KnowledgeGraph(store)
    binder = EvidenceBinder(graph, EvidenceLedger(store))
    yield store, graph, ProposalBoard(store, binder)
    store.close()


def package(contributor: str, outcome: str = "supported", **extra) -> EvidencePackage:
    return EvidencePackage.build(
        claim="converter exceeds 80% efficiency",
        concept=CONCEPT,
        contributor=contributor,
        outcome=outcome,
        **extra,
    )


def test_lifecycle_open_claim_resolve(env) -> None:
    store, graph, board = env
    proposal = board.propose(
        CONCEPT, "Does the converter exceed 80%?", rationale="planner gap"
    )
    assert proposal.status == "open"

    claimed = board.claim(proposal.id, "alice")
    assert claimed.status == "claimed"
    assert claimed.claimed_by == "alice"

    original = package("alice")
    resolved = board.resolve(
        proposal.id,
        [original, package("bob", kind="replication", replicates=original.id)],
    )
    assert resolved.status == "resolved"
    assert resolved.resolution.outcome == "supported"
    assert len(resolved.resolution.package_ids) == 2
    # the evidence flowed into the graph
    assert graph.get(CONCEPT).confidence > 0.6
    # full lifecycle history preserved
    assert len(store.history("proposals", proposal.id)) == 3


def test_duplicate_open_proposal_returns_existing(env) -> None:
    _, _, board = env
    first = board.propose(CONCEPT, "Does it work?", rationale="a")
    second = board.propose(CONCEPT, "Does it work?", rationale="b")
    assert first.id == second.id
    assert len(board.proposals()) == 1


def test_transition_guards(env) -> None:
    _, _, board = env
    proposal = board.propose(CONCEPT, "Does it work?", rationale="x")
    board.claim(proposal.id, "alice")
    with pytest.raises(ProposalError, match="not open"):
        board.claim(proposal.id, "bob")
    board.resolve(proposal.id, [package("alice")])
    with pytest.raises(ProposalError, match="already resolved"):
        board.resolve(proposal.id, [package("bob")])
    with pytest.raises(ProposalError, match="unknown"):
        board.claim("prop_nope", "alice")


def test_resolution_requires_relevant_packages(env) -> None:
    _, _, board = env
    proposal = board.propose(CONCEPT, "Does it work?", rationale="x")
    stray = EvidencePackage.build(
        claim="unrelated", concept="Something Else", contributor="a", outcome="supported"
    )
    with pytest.raises(ProposalError, match="targets concept"):
        board.resolve(proposal.id, [stray])


def test_challenging_evidence_resolves_challenged(env) -> None:
    _, graph, board = env
    proposal = board.propose(CONCEPT, "Does it work?", rationale="x")
    resolved = board.resolve(
        proposal.id,
        [package("alice", outcome="challenged"), package("bob", outcome="challenged")],
    )
    assert resolved.resolution.outcome == "challenged"
    assert graph.get(CONCEPT).confidence < 0.5


def test_from_debate_and_conflict_factories(env) -> None:
    _, _, board = env
    question = Question(id="q", prompt="Does the converter self-heat?", topic=CONCEPT)
    result = DebateEngine(disagreement_threshold=0.4).debate(
        question,
        [
            ScriptedStudent("a", "x", knowledge={question.prompt: "yes"}),
            ScriptedStudent("b", "x", knowledge={question.prompt: "no"}),
        ],
    )
    from_debate = board.from_debate(result)
    assert from_debate.origin == "debate"
    assert from_debate.concept == CONCEPT

    conflict = ConflictNode(
        concept=CONCEPT,
        interpretation_a="it is a heat pump",
        interpretation_b="it is a turbine",
        sources=("doc_1", "doc_2"),
        evidence=(SpanRef(doc="doc_1", start=0, end=10),),
    )
    from_conflict = board.from_conflict(conflict)
    assert from_conflict.origin == "conflict"
    assert "Which interpretation" in from_conflict.question
    assert len(board.proposals(status="open")) == 2


def test_proposals_filter_by_status(env) -> None:
    _, _, board = env
    keep_open = board.propose(CONCEPT, "q1?", rationale="x")
    to_resolve = board.propose(CONCEPT, "q2?", rationale="x")
    board.resolve(to_resolve.id, [package("alice")])
    assert [p.id for p in board.proposals(status="open")] == [keep_open.id]
    assert [p.id for p in board.proposals(status="resolved")] == [to_resolve.id]
