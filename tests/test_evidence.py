"""Tests for allm.evidence: packages, replication-aware confidence, binding."""

from pathlib import Path

import pytest

from allm.evidence import (
    EvidenceBinder,
    EvidenceLedger,
    EvidencePackage,
    evidential_confidence,
)
from allm.knowledge import KnowledgeGraph
from allm.storage import SQLiteRecordStore

CONCEPT = "High-Efficiency Energy Converter"


def package(
    contributor: str,
    outcome: str = "supported",
    kind: str = "experiment",
    replicates: str | None = None,
    claim: str = "converter exceeds 80% efficiency",
    **extra,
) -> EvidencePackage:
    return EvidencePackage.build(
        claim=claim,
        concept=CONCEPT,
        contributor=contributor,
        kind=kind,
        outcome=outcome,
        replicates=replicates,
        **extra,
    )


# -- confidence model ----------------------------------------------------


def test_no_packages_is_unmeasured() -> None:
    assert evidential_confidence(CONCEPT, []) is None


def test_single_experiment_earns_moderate_confidence() -> None:
    breakdown = evidential_confidence(CONCEPT, [package("alice")])
    assert breakdown is not None
    assert 0.55 < breakdown.value < 0.7  # positive but far from certain


def test_popularity_does_not_move_confidence() -> None:
    once = evidential_confidence(CONCEPT, [package("alice")])
    spammed = evidential_confidence(
        CONCEPT,
        [package("alice", claim=f"converter run {i} works") for i in range(50)],
    )
    assert spammed.value == once.value  # 50 posts by one lab = 1 post


def test_independent_replication_beats_repetition() -> None:
    original = package("alice")
    replication = package("bob", kind="replication", replicates=original.id)
    replicated = evidential_confidence(CONCEPT, [original, replication])
    solo = evidential_confidence(CONCEPT, [original])
    assert replicated.value > solo.value
    assert replicated.independent_replications == 1


def test_self_replication_downgraded_to_experiment() -> None:
    original = package("alice")
    self_rep = package("alice", kind="replication", replicates=original.id,
                       claim="my own re-run")
    breakdown = evidential_confidence(CONCEPT, [original, self_rep])
    assert breakdown.independent_replications == 0
    solo = evidential_confidence(CONCEPT, [original])
    assert breakdown.value == solo.value  # same contributor, same stance: capped


def test_failed_replication_drags_confidence_down() -> None:
    original = package("alice")
    supported = evidential_confidence(CONCEPT, [original])
    failed = package("carol", kind="replication", outcome="challenged",
                     replicates=original.id)
    contested = evidential_confidence(CONCEPT, [original, failed])
    assert contested.value < supported.value
    assert contested.value < 0.5  # independent challenge outweighs one experiment


def test_inconclusive_adds_uncertainty_not_stance() -> None:
    original = package("alice")
    inconclusive = package("dave", outcome="inconclusive")
    breakdown = evidential_confidence(CONCEPT, [original, inconclusive])
    solo = evidential_confidence(CONCEPT, [original])
    assert breakdown.value < solo.value  # denominator grew
    assert breakdown.challenge_weight == 0.0


def test_breakdown_is_fully_transparent() -> None:
    original = package("alice")
    breakdown = evidential_confidence(CONCEPT, [original])
    assert breakdown.packages == (original.id,)
    assert breakdown.contributors == 1


def test_build_ids_are_deterministic() -> None:
    assert package("alice").id == package("alice").id
    assert package("alice").id != package("bob").id


# -- ledger + binder --------------------------------------------------------


@pytest.fixture()
def env(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "evidence.sqlite3")
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    yield graph, ledger, EvidenceBinder(graph, ledger)
    store.close()


def test_submit_creates_concept_with_provenance(env) -> None:
    graph, ledger, binder = env
    breakdown = binder.submit(package("alice"))
    concept = graph.get(CONCEPT)
    assert concept is not None
    assert concept.source == "evidence"
    assert concept.confidence == breakdown.value
    assert concept.evidence[0].source == ledger.packages_for(CONCEPT)[0].id


def test_submit_revises_and_recomputes(env) -> None:
    graph, ledger, binder = env
    original = package("alice")
    binder.submit(original)
    first_confidence = graph.get(CONCEPT).confidence
    binder.submit(package("bob", kind="replication", replicates=original.id))
    concept = graph.get(CONCEPT)
    assert concept.confidence > first_confidence
    assert len(concept.evidence) == 2
    assert len(graph.history(CONCEPT)) == 2  # versioned, never overwritten


def test_challenging_evidence_is_marked_non_supporting(env) -> None:
    graph, _, binder = env
    binder.submit(package("alice", outcome="challenged"))
    assert not graph.get(CONCEPT).evidence[0].supports


def test_why_renders_provenance_tree(env) -> None:
    graph, _, binder = env
    original = package("alice")
    binder.submit(original)
    binder.submit(package("bob", kind="replication", replicates=original.id))
    binder.submit(package("carol", kind="replication", outcome="challenged",
                          replicates=original.id))
    tree = binder.why(CONCEPT)
    assert CONCEPT in tree
    assert "[+] experiment" in tree
    assert "[+] replication" in tree
    assert "[-] replication" in tree
    assert binder.why("Unknown Concept").endswith("no evidence packages recorded")
