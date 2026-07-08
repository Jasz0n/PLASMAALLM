"""The knowledge graph: versioned concepts over the record store.

Design decisions
----------------
- Nodes are records in namespace ``concepts``; edges live *on* the
  nodes (prerequisite and related name lists). At research scale this
  keeps the store schema-agnostic and the whole graph greppable; a
  dedicated graph index can be added later behind the same class.
- All mutation goes through :meth:`add` and :meth:`revise`. ``revise``
  builds a new version and *structurally cannot drop evidence*: the new
  evidence tuple must extend the old one (Plan.md: never lose
  supporting evidence; compression in Phase 9 must carry evidence over,
  not delete it).
- Cycles in prerequisites are rejected at write time — a curriculum
  with circular prerequisites can never be scheduled by the planner.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable

from allm.core.logging import get_logger
from allm.knowledge.types import Concept, Evidence
from allm.planner.signals import TopicInfo
from allm.storage.base import RecordStore

logger = get_logger("knowledge.graph")

NAMESPACE = "concepts"


class KnowledgeGraphError(ValueError):
    """Raised for structurally invalid graph mutations."""


class KnowledgeGraph:
    """Versioned concept graph."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    # -- reads ----------------------------------------------------------

    def get(self, name: str) -> Concept | None:
        record = self._store.get(NAMESPACE, name)
        return None if record is None else Concept.model_validate(record.value)

    def history(self, name: str) -> list[Concept]:
        """Every recorded version of a concept, oldest first."""
        return [
            Concept.model_validate(r.value) for r in self._store.history(NAMESPACE, name)
        ]

    def names(self) -> list[str]:
        return self._store.keys(NAMESPACE)

    def concepts(self) -> list[Concept]:
        return [c for name in self.names() if (c := self.get(name)) is not None]

    def dependents_of(self, name: str) -> list[str]:
        """Concepts that list ``name`` as a prerequisite."""
        return sorted(c.name for c in self.concepts() if name in c.prerequisites)

    def neighbours(self, name: str) -> list[str]:
        """Union of prerequisites, dependents and related concepts."""
        concept = self.get(name)
        if concept is None:
            return []
        linked = set(concept.prerequisites) | set(concept.related)
        linked.update(self.dependents_of(name))
        return sorted(linked)

    # -- writes ---------------------------------------------------------

    def add(self, concept: Concept, *, reason: str = "initial") -> Concept:
        """Insert a new concept; fails if it exists or creates a cycle."""
        if self.get(concept.name) is not None:
            raise KnowledgeGraphError(
                f"concept {concept.name!r} exists; use revise() to change it"
            )
        self._check_acyclic(concept.name, concept.prerequisites)
        self._put(concept, reason)
        logger.info("added concept %r (%s)", concept.name, reason)
        return concept

    def revise(
        self,
        name: str,
        *,
        reason: str,
        description: str | None = None,
        confidence: float | None = None,
        usefulness: float | None = None,
        curiosity: float | None = None,
        status: str | None = None,
        add_prerequisites: Iterable[str] = (),
        add_related: Iterable[str] = (),
        add_evidence: Iterable[Evidence] = (),
    ) -> Concept:
        """Create a new version of ``name``. Only additive for edges and
        evidence; scalar fields may be re-estimated. ``reason`` is
        mandatory — Plan.md requires knowing why beliefs changed.
        """
        current = self.get(name)
        if current is None:
            raise KnowledgeGraphError(f"unknown concept {name!r}; add() it first")
        prerequisites = _extend(current.prerequisites, add_prerequisites)
        self._check_acyclic(name, prerequisites)
        revised = current.model_copy(
            update={
                "description": description if description is not None else current.description,
                "confidence": confidence if confidence is not None else current.confidence,
                "usefulness": usefulness if usefulness is not None else current.usefulness,
                "curiosity": curiosity if curiosity is not None else current.curiosity,
                "status": status if status is not None else current.status,
                "prerequisites": prerequisites,
                "related": _extend(current.related, add_related),
                "evidence": current.evidence + tuple(add_evidence),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._put(revised, reason)
        logger.info("revised concept %r (%s)", name, reason)
        return revised

    # -- planner bridge ---------------------------------------------------

    def to_catalog(self) -> dict[str, TopicInfo]:
        """Export the graph as a planner topic catalog.

        Usefulness maps to importance; prerequisites to dependencies.
        Retracted concepts are excluded — they are history, not
        curriculum. This replaces the hand-written YAML catalog once a
        graph exists.
        """
        return {
            c.name: TopicInfo(
                importance=c.usefulness,
                curiosity=c.curiosity,
                dependencies=c.prerequisites,
            )
            for c in self.concepts()
            if c.status == "active"
        }

    # -- internals --------------------------------------------------------

    def _put(self, concept: Concept, reason: str) -> None:
        self._store.put(
            NAMESPACE, concept.name, json.loads(concept.model_dump_json()), reason=reason
        )

    def _check_acyclic(self, name: str, prerequisites: tuple[str, ...]) -> None:
        """Reject prerequisite chains that lead back to ``name``."""
        stack, seen = list(prerequisites), set()
        while stack:
            current = stack.pop()
            if current == name:
                raise KnowledgeGraphError(
                    f"prerequisite cycle: {name!r} would depend on itself"
                )
            if current in seen:
                continue
            seen.add(current)
            concept = self.get(current)
            if concept is not None:
                stack.extend(concept.prerequisites)


def _extend(current: tuple[str, ...], additions: Iterable[str]) -> tuple[str, ...]:
    merged = list(current)
    for item in additions:
        if item not in merged:
            merged.append(item)
    return tuple(merged)
