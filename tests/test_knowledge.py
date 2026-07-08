"""Tests for allm.knowledge: versioned concepts, evidence, cycles, catalog."""

from pathlib import Path

import pytest

from allm.knowledge import Concept, Evidence, KnowledgeGraph, KnowledgeGraphError
from allm.storage import SQLiteRecordStore


@pytest.fixture()
def graph(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "kg.sqlite3")
    yield KnowledgeGraph(store)
    store.close()


def gravity() -> Concept:
    return Concept(
        name="gravity",
        description="masses attract",
        confidence=0.6,
        source="textbook",
        evidence=(Evidence(source="apple", detail="it fell"),),
    )


def test_add_and_get(graph: KnowledgeGraph) -> None:
    graph.add(gravity())
    concept = graph.get("gravity")
    assert concept.description == "masses attract"
    assert concept.evidence[0].source == "apple"
    assert graph.get("phlogiston") is None


def test_add_duplicate_rejected(graph: KnowledgeGraph) -> None:
    graph.add(gravity())
    with pytest.raises(KnowledgeGraphError, match="revise"):
        graph.add(gravity())


def test_revise_versions_and_keeps_evidence(graph: KnowledgeGraph) -> None:
    graph.add(gravity())
    graph.revise(
        "gravity",
        reason="passed the mechanics exam",
        confidence=0.9,
        add_evidence=[Evidence(source="exam-0001", detail="scored 1.0")],
    )
    latest = graph.get("gravity")
    assert latest.confidence == 0.9
    assert [e.source for e in latest.evidence] == ["apple", "exam-0001"]
    history = graph.history("gravity")
    assert [c.confidence for c in history] == [0.6, 0.9]
    assert latest.updated_at >= history[0].updated_at
    assert latest.learned_at == history[0].learned_at


def test_revise_unknown_rejected(graph: KnowledgeGraph) -> None:
    with pytest.raises(KnowledgeGraphError, match="add"):
        graph.revise("phlogiston", reason="oops")


def test_edges_are_additive_and_deduplicated(graph: KnowledgeGraph) -> None:
    graph.add(Concept(name="algebra"))
    graph.add(Concept(name="calculus", prerequisites=("algebra",)))
    graph.revise("calculus", reason="link", add_prerequisites=["algebra"], add_related=["physics"])
    concept = graph.get("calculus")
    assert concept.prerequisites == ("algebra",)
    assert concept.related == ("physics",)


def test_dependents_and_neighbours(graph: KnowledgeGraph) -> None:
    graph.add(Concept(name="algebra"))
    graph.add(Concept(name="calculus", prerequisites=("algebra",), related=("physics",)))
    assert graph.dependents_of("algebra") == ["calculus"]
    assert graph.neighbours("calculus") == ["algebra", "physics"]
    assert graph.neighbours("algebra") == ["calculus"]


def test_direct_cycle_rejected(graph: KnowledgeGraph) -> None:
    with pytest.raises(KnowledgeGraphError, match="cycle"):
        graph.add(Concept(name="a", prerequisites=("a",)))


def test_transitive_cycle_rejected(graph: KnowledgeGraph) -> None:
    graph.add(Concept(name="a"))
    graph.add(Concept(name="b", prerequisites=("a",)))
    with pytest.raises(KnowledgeGraphError, match="cycle"):
        graph.revise("a", reason="loop", add_prerequisites=["b"])


def test_to_catalog_maps_fields(graph: KnowledgeGraph) -> None:
    graph.add(Concept(name="algebra", usefulness=0.9, curiosity=0.2))
    graph.add(Concept(name="calculus", prerequisites=("algebra",), curiosity=0.8))
    catalog = graph.to_catalog()
    assert catalog["algebra"].importance == 0.9
    assert catalog["calculus"].dependencies == ("algebra",)
    assert catalog["calculus"].curiosity == 0.8
