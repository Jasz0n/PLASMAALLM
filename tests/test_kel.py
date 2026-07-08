"""Tests for allm.kel: each metric, the composite, time series, diagnosis."""

from pathlib import Path

import pytest

from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kel import KELConfig, KnowledgeEvaluationLayer
from allm.kel.metrics import (
    concept_reuse,
    conflict_density,
    learning_gain,
    rcr,
    snapshot,
    stability,
)
from allm.knowledge import Concept, KnowledgeGraph
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig


@pytest.fixture()
def env(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "kel.sqlite3")
    graph = KnowledgeGraph(store)
    state = KnowledgeState(store)
    yield store, graph, state
    store.close()


def distill(*texts: str):
    documents = DocumentStore()
    for i, text in enumerate(texts):
        documents.ingest_text(f"t{i}.md", text)
    return KDPipeline().distill(documents)


def teach(store, graph, *, learn: bool) -> Teacher:
    """Examine a student on graph-derived topics; optionally let it learn."""
    facts = {"2+2?": ("4", "math"), "Capital of France?": ("Paris", "geography")}
    samples = [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": t})
        for i, (q, (a, t)) in enumerate(facts.items())
    ]
    teacher = Teacher(
        KnowledgeState(store),
        DatasetExamGenerator(samples),
        ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=1.0),
    )
    student = ScriptedStudent("kid", "general")
    teacher.evaluate(student, teacher.create_exam(num_questions=2, seed=1))
    if learn:
        for question, (answer, _) in facts.items():
            student.learn(question, answer)
        teacher.evaluate(student, teacher.create_exam(num_questions=2, seed=2))
    return teacher


# -- individual metrics --------------------------------------------------


def test_rcr_measures_collapse() -> None:
    result = distill(
        "Gravity is the attraction between masses.",
        "Gravity is the mutual attraction of masses.",
    )
    value = rcr(result)
    assert value is not None and 0.0 < value < 1.0  # 2 mentions -> 1 concept


def test_conflict_density(env) -> None:
    store, graph, _ = env
    assert conflict_density(graph, store) is None  # empty graph: unmeasurable
    graph.add(Concept(name="a"))
    graph.add(Concept(name="b"))
    assert conflict_density(graph, store) == 0.0
    store.put("conflicts", "a#0", {"concept": "a"}, reason="test")
    assert conflict_density(graph, store) == 0.5


def test_stability_of_identical_and_changed_graphs(env) -> None:
    _, graph, _ = env
    graph.add(Concept(name="a"))
    first = snapshot(graph)
    assert stability(first, snapshot(graph)) == 1.0
    graph.add(Concept(name="b", prerequisites=("a",)))
    assert stability(first, snapshot(graph)) < 1.0


def test_retracted_concepts_leave_the_snapshot(env) -> None:
    _, graph, _ = env
    graph.add(Concept(name="a"))
    graph.add(Concept(name="b"))
    before = snapshot(graph)
    graph.revise("b", reason="test retraction", status="retracted")
    assert "b" in before.nodes
    assert "b" not in snapshot(graph).nodes


def test_concept_reuse_counts_exams_and_goals(env) -> None:
    store, graph, _ = env
    graph.add(Concept(name="math"))
    graph.add(Concept(name="geography"))
    graph.add(Concept(name="never-used"))
    teacher = teach(store, graph, learn=False)
    teacher.assign_goals("kid")
    value = concept_reuse(graph, teacher.state)
    # 2 exam questions + goals, spread over 3 concepts
    assert value is not None and value > 0.5


def test_learning_gain_only_over_graph_concepts(env) -> None:
    store, graph, _ = env
    graph.add(Concept(name="math"))  # geography deliberately not in the graph
    teach(store, graph, learn=True)
    assert learning_gain(graph, KnowledgeState(store)) == 1.0  # 0 -> 1 on math


# -- the layer: evaluate, GHS, series, trends --------------------------------


def test_first_evaluation_has_no_gst_or_ghs(env) -> None:
    store, graph, state = env
    graph.add(Concept(name="math"))
    kel = KnowledgeEvaluationLayer(graph, store, state)
    report = kel.evaluate(distill("Math is the study of structure."))
    assert report.gst is None  # nothing to compare against yet
    assert report.ghs is None  # composite needs every component


def test_second_evaluation_computes_gst_and_ghs(env) -> None:
    store, graph, state = env
    result = distill(
        "Dropout is a regularisation technique that removes random units.",
        "Dropout is the compression of gradients for faster syncing.",
        "Math is the study of structure. Math is always abstract.",
    )
    GraphInjector(graph, store).inject(result)
    graph.add(Concept(name="math"))
    graph.add(Concept(name="geography"))
    teacher = teach(store, graph, learn=True)

    kel = KnowledgeEvaluationLayer(graph, store, teacher.state)
    kel.evaluate(result)
    report = kel.evaluate()  # rcr carries forward, gst now measurable
    assert report.gst == 1.0  # graph unchanged between evaluations
    assert report.cre is not None  # conflicts exist -> measurable
    assert report.ghs is not None
    assert len(kel.history("cd")) == 2
    assert kel.trend("cd") == 0.0


def test_trend_requires_two_points(env) -> None:
    store, graph, state = env
    kel = KnowledgeEvaluationLayer(graph, store, state)
    assert kel.trend("lg") is None


# -- failure modes ---------------------------------------------------------------


def put_metric(store, metric: str, *values: float) -> None:
    for value in values:
        store.put("kel_metrics", metric, {"value": value}, reason="test")


def test_false_compression_detected(env) -> None:
    store, graph, state = env
    put_metric(store, "rcr", 0.8)
    put_metric(store, "crr", 0.2)
    kel = KnowledgeEvaluationLayer(graph, store, state)
    assert [f.mode for f in kel.diagnose()] == ["false_compression"]


def test_conflict_accumulation_detected(env) -> None:
    store, graph, state = env
    put_metric(store, "cd", 0.5)
    put_metric(store, "cre", 0.1)
    kel = KnowledgeEvaluationLayer(graph, store, state)
    assert "conflict_accumulation" in [f.mode for f in kel.diagnose()]


def test_static_illusion_detected(env) -> None:
    store, graph, state = env
    put_metric(store, "gst", 0.95)
    put_metric(store, "lg", 0.5, 0.2)  # declining
    kel = KnowledgeEvaluationLayer(graph, store, state)
    assert "static_illusion" in [f.mode for f in kel.diagnose()]


def test_dead_knowledge_growth_detected(env) -> None:
    store, graph, state = env
    put_metric(store, "crr", 0.3)
    store.put("kel_snapshots", "graph", {"nodes": ["a"], "edges": []}, reason="t")
    store.put(
        "kel_snapshots", "graph", {"nodes": ["a", "b", "c"], "edges": []}, reason="t"
    )
    kel = KnowledgeEvaluationLayer(graph, store, state)
    assert "dead_knowledge_growth" in [f.mode for f in kel.diagnose()]


def test_healthy_system_has_no_findings(env) -> None:
    store, graph, state = env
    put_metric(store, "rcr", 0.4)
    put_metric(store, "crr", 3.0)
    put_metric(store, "cd", 0.1)
    put_metric(store, "cre", 0.9)
    put_metric(store, "gst", 0.8)
    put_metric(store, "lg", 0.1, 0.3)
    kel = KnowledgeEvaluationLayer(graph, store, state)
    assert kel.diagnose() == []


def test_kel_config_is_tunable(env) -> None:
    store, graph, state = env
    put_metric(store, "rcr", 0.4)
    put_metric(store, "crr", 0.2)
    strict = KnowledgeEvaluationLayer(graph, store, state, KELConfig(high_rcr=0.3))
    assert [f.mode for f in strict.diagnose()] == ["false_compression"]
