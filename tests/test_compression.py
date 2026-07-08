"""Tests for allm.compression: proposals, evidence preservation, retraction."""

from pathlib import Path

import pytest

from allm.compression import CompressionEngine, PerformanceProbe
from allm.knowledge import Concept, Evidence, KnowledgeGraph
from allm.storage import SQLiteRecordStore


class FakeProbe:
    """Probe returning scripted scores in order."""

    def __init__(self, *scores: float) -> None:
        self._scores = list(scores)

    def score(self) -> float:
        return self._scores.pop(0)


@pytest.fixture()
def graph(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "kg.sqlite3")
    yield KnowledgeGraph(store)
    store.close()


def seed_shared_foundation(graph: KnowledgeGraph) -> None:
    graph.add(Concept(name="newton-laws"))
    for name in ("orbits", "tides", "falling-apples"):
        graph.add(
            Concept(
                name=name,
                prerequisites=("newton-laws",),
                confidence=0.7,
                evidence=(Evidence(source=f"{name}-observation"),),
            )
        )


def test_propose_groups_identical_prerequisites(graph: KnowledgeGraph) -> None:
    seed_shared_foundation(graph)
    graph.add(Concept(name="unrelated"))  # no prerequisites -> never grouped
    proposals = CompressionEngine(graph).propose()
    assert len(proposals) == 1
    assert proposals[0].members == ("falling-apples", "orbits", "tides")
    assert proposals[0].shared_prerequisites == ("newton-laws",)


def test_min_group_respected(graph: KnowledgeGraph) -> None:
    graph.add(Concept(name="base"))
    graph.add(Concept(name="one", prerequisites=("base",)))
    graph.add(Concept(name="two", prerequisites=("base",)))
    assert CompressionEngine(graph, min_group=3).propose() == []
    assert len(CompressionEngine(graph, min_group=2).propose()) == 1


def test_apply_preserves_members_and_evidence(graph: KnowledgeGraph) -> None:
    seed_shared_foundation(graph)
    engine = CompressionEngine(graph)
    outcome = engine.apply(engine.propose()[0])
    assert outcome.applied and not outcome.retracted

    principle = graph.get(outcome.proposal.principle)
    assert principle.source == "compression"
    assert {e.source for e in principle.evidence} == {
        "orbits-observation", "tides-observation", "falling-apples-observation"
    }
    assert principle.confidence == 0.7  # min of members
    # members survive and now point at the principle
    for member in outcome.proposal.members:
        assert graph.get(member) is not None
        assert outcome.proposal.principle in graph.get(member).related


def test_probe_regression_retracts_principle(graph: KnowledgeGraph) -> None:
    seed_shared_foundation(graph)
    engine = CompressionEngine(graph, probe=FakeProbe(0.9, 0.7), tolerance=0.05)
    outcome = engine.apply(engine.propose()[0])
    assert outcome.retracted
    principle = graph.get(outcome.proposal.principle)
    assert principle.status == "retracted"
    assert principle.evidence  # evidence survives retraction
    assert principle.name not in graph.to_catalog()  # not curriculum anymore
    history = graph.history(principle.name)
    assert history[0].status == "active"  # full story preserved


def test_probe_within_tolerance_keeps_principle(graph: KnowledgeGraph) -> None:
    seed_shared_foundation(graph)
    engine = CompressionEngine(graph, probe=FakeProbe(0.9, 0.88), tolerance=0.05)
    outcome = engine.apply(engine.propose()[0])
    assert not outcome.retracted
    assert graph.get(outcome.proposal.principle).status == "active"


def test_compress_is_idempotent(graph: KnowledgeGraph) -> None:
    seed_shared_foundation(graph)
    engine = CompressionEngine(graph)
    first = engine.compress()
    second = engine.compress()
    assert len(first) == 1
    assert second == []  # existing principle is not re-proposed


def test_probe_protocol() -> None:
    assert isinstance(FakeProbe(1.0), PerformanceProbe)
