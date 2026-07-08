"""Evidence Growth Rate + unearned confidence (KEL.md 3.7 / 9.5)."""

from pathlib import Path

import pytest

from allm.evidence import EvidenceLedger, EvidencePackage
from allm.kel import KnowledgeEvaluationLayer
from allm.kel.metrics import evidence_foundation, evidence_growth
from allm.knowledge import Concept, KnowledgeGraph
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState


def package(concept: str, contributor: str, kind: str = "experiment", **kwargs):
    return EvidencePackage.build(
        claim=f"claim about {concept} by {contributor} ({kind})",
        concept=concept,
        contributor=contributor,
        kind=kind,
        outcome="supported",
        **kwargs,
    )


def test_foundation_rewards_diversity_and_replication() -> None:
    one_lab_spam = [package("coating", "lab-a") for _ in range(3)]
    # identical claims content-address to the same id, so vary them
    one_lab_spam = [
        package("coating", "lab-a", measurements={"n": i}) for i in range(50)
    ]
    diverse = [
        package("coating", "lab-a"),
        package("coating", "lab-b", kind="paper"),
        package("coating", "lab-c", kind="observation"),
        package("coating", "lab-d", kind="replication", replicates="pkg_x"),
        package("coating", "lab-e", kind="replication", replicates="pkg_y"),
    ]
    # 50 posts from one lab: 50 packages + 1 kind + 1 contributor = 52
    # 5 diverse contributions: 5 + 4 kinds + 5 contributors + 2*2 = 18
    assert evidence_foundation(one_lab_spam) == 52.0
    assert evidence_foundation(diverse) == 18.0
    # per *marginal* contribution, diversity is worth far more
    assert evidence_foundation(diverse) / 5 > evidence_foundation(one_lab_spam) / 50


def test_growth_is_relative_and_none_on_first_measurement() -> None:
    assert evidence_growth(None, 10.0) is None
    assert evidence_growth(10.0, 15.0) == 0.5
    assert evidence_growth(0.0, 7.0) == 7.0  # bootstrap: relative to max(prev, 1)
    assert evidence_growth(10.0, 10.0) == 0.0


@pytest.fixture()
def env(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "kel.sqlite3")
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    kel = KnowledgeEvaluationLayer(graph, store, KnowledgeState(store), ledger=ledger)
    yield graph, ledger, kel
    store.close()


def test_egr_tracks_earned_evidence(env) -> None:
    graph, ledger, kel = env
    graph.add(Concept(name="coating"))
    first = kel.evaluate()
    assert first.egr is None  # first measurement: cannot measure ≠ zero

    second = kel.evaluate()
    assert second.egr == 0.0  # nothing earned between measurements

    ledger.submit(package("coating", "lab-a"))
    ledger.submit(package("coating", "lab-b", kind="replication", replicates="pkg_x"))
    third = kel.evaluate()
    assert third.egr is not None and third.egr > 0

    # without a ledger the metric is honestly unmeasurable
    no_ledger = KnowledgeEvaluationLayer(graph, kel._store, kel._state)
    assert no_ledger.evaluate().egr is None


def test_unearned_confidence_detected_and_cleared(env) -> None:
    graph, ledger, kel = env
    graph.add(Concept(name="hearsay-claim", confidence=0.9, source="kdp"))
    graph.add(Concept(name="modest-claim", confidence=0.4, source="kdp"))
    kel.evaluate()
    findings = {f.mode: f for f in kel.diagnose()}
    assert "unearned_confidence" in findings
    assert "hearsay-claim" in findings["unearned_confidence"].detail
    assert "modest-claim" not in findings["unearned_confidence"].detail

    # evidence arrives: the confidence is earned now
    ledger.submit(package("hearsay-claim", "lab-a"))
    assert not [f for f in kel.diagnose() if f.mode == "unearned_confidence"]
