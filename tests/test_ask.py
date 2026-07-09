"""Ask ALLM — grounded Q&A (M52). It answers from evidence or says 'no'."""

from pathlib import Path

import pytest

from allm.ask import answer_question
from allm.evidence import EvidenceBinder, EvidenceLedger
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard
from allm.seed import seed_public_loop
from allm.storage import SQLiteRecordStore


@pytest.fixture()
def seeded(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "ask.sqlite3")
    seed_public_loop(store)
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    binder = EvidenceBinder(graph, ledger)
    board = ProposalBoard(store, binder)
    return graph, ledger, board, binder


def test_established_claim_is_answered_with_its_evidence(seeded) -> None:
    graph, ledger, board, binder = seeded
    a = answer_question(
        "how long does the nano coating take to form?", graph, ledger, board, binder=binder
    )
    assert a.found and a.status == "established"
    assert a.concept == "The Nano Coating"
    assert "12 hours" in a.answer  # the settled answer, grounded
    assert a.confidence >= 0.75 and a.independent_replications >= 2
    assert a.sources and a.provenance  # traceable


def test_how_to_returns_the_reproduced_procedure_not_the_definition(seeded) -> None:
    graph, ledger, board, binder = seeded
    a = answer_question("how do I make a nano coating?", graph, ledger, board, binder=binder)
    assert a.intent == "how_to" and a.status == "procedure"
    assert a.steps and any("caustic" in s.lower() for s in a.steps)  # the actual steps
    assert "how to make" in a.answer.lower() and "reproduced" in a.answer
    # a quantity question about the same concept is answered differently
    q = answer_question("how long does the nano coating take?", graph, ledger, board, binder=binder)
    assert q.intent == "quantity" and "12 hours" in q.answer and not q.steps


def test_how_to_without_steps_refuses_to_invent_them(seeded) -> None:
    graph, ledger, board, binder = seeded
    # Plasma has evidence but no reproduction steps
    a = answer_question("how do I make plasma?", graph, ledger, board, binder=binder)
    assert a.intent == "how_to" and a.status == "no_procedure"
    assert not a.steps and "won't invent" in a.answer
    assert a.suggestion


def test_a_document_only_claim_is_flagged_unfounded(seeded) -> None:
    graph, ledger, board, binder = seeded
    # "A Gans" came from a document but has no evidence packages
    a = answer_question("what is a gans?", graph, ledger, board, binder=binder)
    assert a.status == "unfounded"
    assert "evidence disposes" in a.answer  # the thesis, stated
    assert a.suggestion  # points to contributing


def test_unknown_question_refuses_to_guess(seeded) -> None:
    graph, ledger, board, binder = seeded
    a = answer_question(
        "how do I build a fusion reactor at home?", graph, ledger, board, binder=binder
    )
    assert not a.found and a.status == "unknown" and a.concept is None
    assert "don't have evidence" in a.answer and a.confidence is None
    assert a.suggestion


def test_ask_endpoint_and_chat_page(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_app
    from allm.api.security import AllowAllVerifier

    app = create_app(tmp_path / "api.sqlite3", verifier=AllowAllVerifier())
    with TestClient(app) as client:
        seed_public_loop(SQLiteRecordStore(tmp_path / "api.sqlite3"))

        # /ask is an open read and never guesses
        answer = client.get("/ask", params={"q": "what is the nano coating?"}).json()
        assert answer["found"] and answer["concept"] == "The Nano Coating"
        assert client.get("/ask", params={"q": "zxqw nonsense"}).json()["status"] == "unknown"

        # the chat page is self-contained and subpath-robust
        page = client.get("/chat")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "Ask ALLM" in page.text and "location.pathname" in page.text
        assert "http://" not in page.text and "https://" not in page.text


def test_client_ask(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from allm.api.app import create_app
    from allm.api.security import AllowAllVerifier
    from allm.client import AllmClient, Response

    db = tmp_path / "api.sqlite3"
    app = create_app(db, verifier=AllowAllVerifier())

    class _T:
        def __init__(self, app):
            from fastapi.testclient import TestClient

            self._tc = TestClient(app)

        def request(self, method, url, headers, body):
            r = self._tc.request(method, url, headers=headers, content=body)
            return Response(r.status_code, r.content, dict(r.headers))

    seed_public_loop(SQLiteRecordStore(db))
    allm = AllmClient("http://testserver", transport=_T(app))
    answer = allm.ask("what is plasma?")
    assert answer["concept"] == "Plasma" and answer["confidence"] is not None
