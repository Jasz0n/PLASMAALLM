"""The seven KEL metrics — pure functions over derived data.

Every function reads only what the system already records (graph,
conflicts store, teacher state, KDP results) and returns ``None`` when
its source has no data yet: "cannot measure" must never masquerade as
"measured as zero" (KEL.md section 2: comparability and stability of
metrics come first).
"""

from __future__ import annotations

from allm.kdp.pipeline import DistillationResult
from allm.knowledge.graph import NAMESPACE as CONCEPT_NAMESPACE
from allm.knowledge.graph import KnowledgeGraph
from allm.storage.base import RecordStore
from allm.teacher.state import KnowledgeState
from allm.kel.types import GraphSnapshot

CONFLICT_NAMESPACE = "conflicts"


# -- 3.1 Redundancy Collapse Ratio -------------------------------------


def rcr(result: DistillationResult) -> float | None:
    """1 - unique_concepts / total_raw_concept_mentions."""
    if result.raw_units == 0:
        return None
    unique = len({u.normalized_concept for u in result.units})
    return round(1.0 - unique / result.raw_units, 4)


# -- 3.2 Conflict Density ------------------------------------------------


def conflict_density(graph: KnowledgeGraph, store: RecordStore) -> float | None:
    """conflict nodes / active concepts, clamped to [0, 1]."""
    concepts = [c for c in graph.concepts() if c.status == "active"]
    if not concepts:
        return None
    conflicts = len(store.keys(CONFLICT_NAMESPACE))
    return round(min(1.0, conflicts / len(concepts)), 4)


# -- 3.3 Graph Stability Over Time ----------------------------------------


def snapshot(graph: KnowledgeGraph) -> GraphSnapshot:
    """Structural fingerprint: active nodes + prerequisite/relation edges."""
    nodes = []
    edges = []
    for concept in graph.concepts():
        if concept.status != "active":
            continue
        nodes.append(concept.name)
        edges.extend(("prerequisite", concept.name, p) for p in concept.prerequisites)
        edges.extend(("related", concept.name, r) for r in concept.related)
    return GraphSnapshot(
        nodes=tuple(sorted(nodes)),
        edges=tuple(sorted((k, a, b) for k, a, b in edges)),
    )


def stability(previous: GraphSnapshot, current: GraphSnapshot) -> float:
    """Jaccard similarity of the two structural fingerprints."""
    prev_items = set(previous.nodes) | set(previous.edges)
    curr_items = set(current.nodes) | set(current.edges)
    if not prev_items and not curr_items:
        return 1.0
    return round(len(prev_items & curr_items) / len(prev_items | curr_items), 4)


# -- 3.4 Concept Reuse Rate --------------------------------------------------


def concept_reuse(graph: KnowledgeGraph, state: KnowledgeState) -> float | None:
    """Mean downstream uses per active concept.

    A "use" is an exam question on the concept's topic or an assigned
    learning goal targeting it — the two downstream consumers the
    system records today. Debate/student-task usage joins the count
    once those record per-topic usage.
    """
    concepts = [c.name for c in graph.concepts() if c.status == "active"]
    if not concepts:
        return None
    uses = dict.fromkeys(concepts, 0)
    for student in state.students():
        for result in state.exam_results(student):
            for graded in result.results:
                if graded.question.topic in uses:
                    uses[graded.question.topic] += 1
        for goal in state.current_goals(student):
            if goal.topic in uses:
                uses[goal.topic] += 1
    return round(sum(uses.values()) / len(concepts), 4)


# -- 3.5 Learning Gain -----------------------------------------------------------


def learning_gain(graph: KnowledgeGraph, state: KnowledgeState) -> float | None:
    """Mean confidence delta (latest - first) over graph concepts that
    students have actually been examined on. The only direct learning
    metric; everything else measures structure."""
    concepts = {c.name for c in graph.concepts() if c.status == "active"}
    deltas = []
    for student in state.students():
        for topic in state.topics(student):
            if topic not in concepts:
                continue
            history = state.confidence_history(student, topic)
            if history:
                deltas.append(history[-1][1] - history[0][1])
    if not deltas:
        return None
    return round(sum(deltas) / len(deltas), 4)


# -- 3.6 Contradiction Resolution Efficiency ---------------------------------------


def conflict_resolution_efficiency(
    graph: KnowledgeGraph, store: RecordStore, state: KnowledgeState
) -> float | None:
    """Fraction of conflicts that became productive.

    A conflict counts as productive when its concept was (a) examined
    (an exam question on the topic exists), or (b) refined — the
    concept gained graph revisions after the conflict was recorded.
    """
    conflict_keys = store.keys(CONFLICT_NAMESPACE)
    if not conflict_keys:
        return None
    examined_topics = {
        graded.question.topic
        for student in state.students()
        for result in state.exam_results(student)
        for graded in result.results
    }
    productive = 0
    for key in conflict_keys:
        record = store.get(CONFLICT_NAMESPACE, key)
        concept = record.value["concept"]
        if concept in examined_topics:
            productive += 1
            continue
        revisions_after = [
            r
            for r in store.history(CONCEPT_NAMESPACE, concept)
            if r.created_at > record.created_at
        ]
        if revisions_after:
            productive += 1
    return round(productive / len(conflict_keys), 4)


# -- 3.7 Evidence Growth Rate --------------------------------------------


def evidence_foundation(packages: list) -> float:
    """Total evidential foundation across concepts (KEL.md 3.7).

    Per concept: packages + distinct kinds + distinct contributors
    + 2 × independent replications. Diversity and replication count
    more than volume — fifty posts from one lab barely move it.
    """
    per_concept: dict[str, dict[str, set | int]] = {}
    for package in packages:
        row = per_concept.setdefault(
            package.concept,
            {"count": 0, "kinds": set(), "contributors": set(), "replications": 0},
        )
        row["count"] += 1
        row["kinds"].add(package.kind)
        row["contributors"].add(package.contributor)
        if package.replicates is not None:
            row["replications"] += 1
    total = 0.0
    for row in per_concept.values():
        total += (
            row["count"]
            + len(row["kinds"])
            + len(row["contributors"])
            + 2 * row["replications"]
        )
    return round(total, 4)


def evidence_growth(previous_foundation: float | None, current_foundation: float) -> float | None:
    """Relative foundation growth between measurements; None on the first."""
    if previous_foundation is None:
        return None
    return round(
        (current_foundation - previous_foundation) / max(previous_foundation, 1.0), 4
    )
