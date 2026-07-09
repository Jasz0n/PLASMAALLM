"""System dashboard: read-only aggregation + endpoints (M50/M51), offline."""

from pathlib import Path

import pytest

from allm.api.dashboard import system_state
from allm.evidence import EvidenceLedger, EvidencePackage
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.knowledge.types import Concept
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteRecordStore:
    store = SQLiteRecordStore(tmp_path / "dash.sqlite3")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="plasma", confidence=0.8, usefulness=0.9), reason="seed")
    graph.add(Concept(name="fusion", confidence=0.3, usefulness=0.4), reason="seed")
    ledger = EvidenceLedger(store)
    ledger.submit(
        EvidencePackage.build(
            claim="plasma is hot", concept="plasma", contributor="ada",
            kind="experiment", outcome="supported",
        )
    )
    ledger.submit(
        EvidencePackage.build(
            claim="re-run", concept="plasma", contributor="bob",
            kind="replication", outcome="supported", replicates="pkg_prior",
        )
    )
    # two measurements so the KEL time series has a trend to report
    kel = KnowledgeEvaluationLayer(graph, store, KnowledgeState(store), ledger=ledger)
    kel.evaluate()
    kel.evaluate()
    return store


def test_namespaces_census_counts_population(store: SQLiteRecordStore) -> None:
    stats = {s.namespace: s for s in store.namespaces()}
    assert stats["concepts"].keys == 2
    assert stats["evidence_packages"].keys == 2
    # kel wrote a full metric series twice → many records under few keys
    assert stats["kel_metrics"].records > stats["kel_metrics"].keys
    assert all(s.last_write is not None for s in stats.values())


def test_system_state_reflects_the_whole_engine(store: SQLiteRecordStore) -> None:
    state = system_state(store)

    # KEL scorecard: every metric present, measured ones carry a value
    metrics = {m["name"]: m for m in state["kel"]["metrics"]}
    assert set(metrics) == {"ghs", "rcr", "cd", "gst", "crr", "lg", "cre", "egr", "ks"}
    assert metrics["cd"]["measurements"] == 2  # evaluated twice
    assert metrics["cd"]["latest"] is not None
    assert metrics["cd"]["higher_is_better"] is False  # conflict density: less is better

    # graph: buckets place 0.3 and 0.8 in different quartiles
    g = state["graph"]
    assert g["total"] == 2 and g["active"] == 2
    assert g["confidence_buckets"] == [0, 1, 0, 1]
    assert g["top_concepts"][0]["name"] == "plasma"  # highest usefulness first

    # evidence: kinds, outcomes, contributors, replications
    e = state["evidence"]
    assert e["total"] == 2 and e["contributors"] == 2 and e["replications"] == 1
    assert e["by_kind"] == {"experiment": 1, "replication": 1}
    assert e["by_outcome"] == {"supported": 2}

    # census + audit are populated and metadata-only
    assert any(c["namespace"] == "evidence_packages" for c in state["census"])
    assert state["audit"] and "value" not in state["audit"][0]


def test_empty_store_does_not_crash(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "empty.sqlite3")
    try:
        state = system_state(store)
        assert state["graph"]["total"] == 0
        assert state["graph"]["mean_confidence"] is None
        assert state["kel"]["metrics"][0]["latest"] is None
        assert state["census"] == []
        assert state["kel"]["findings"] == []
    finally:
        store.close()


def test_cli_dashboard_state_and_snapshot(tmp_path: Path, store: SQLiteRecordStore, capsys) -> None:
    from allm.cli.main import main

    db = str(tmp_path / "dash.sqlite3")  # the populated fixture store
    assert main(["dashboard", "--db", db]) == 0
    printed = capsys.readouterr().out
    assert '"kel"' in printed and '"census"' in printed  # JSON state to stdout

    snapshot = tmp_path / "snapshot.html"
    assert main(["dashboard", "--db", db, "--output", str(snapshot)]) == 0
    html = snapshot.read_text()
    assert "__ALLM_STATE__" in html  # standalone, state baked in
    assert "http://" not in html and "https://" not in html  # offline-safe


def test_dashboard_endpoints_are_open_reads(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from allm.api.app import create_app
    from allm.api.security import StaticTokenVerifier

    # even with a real verifier configured, reads stay open
    app = create_app(
        tmp_path / "api.sqlite3",
        verifier=StaticTokenVerifier("x" * 16),
    )
    with TestClient(app) as client:
        state = client.get("/dashboard/state")
        assert state.status_code == 200
        assert {"kel", "graph", "evidence", "census", "audit"} <= set(state.json())

        ui = client.get("/dashboard")
        assert ui.status_code == 200
        assert "text/html" in ui.headers["content-type"]
        assert "System Dashboard" in ui.text
        # self-contained: no external resource the CSP/offline box would block
        assert "http://" not in ui.text and "https://" not in ui.text
