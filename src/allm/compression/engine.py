"""Compression engine: fewer, higher-level principles.

Plan.md: "Can five ideas become one? ... Replace the larger
representation with the smaller one while preserving evidence."

Design decisions
----------------
- Compression is *additive abstraction*, never deletion: applying a
  proposal creates a new principle concept whose evidence is the union
  of its members' evidence; members stay in the graph, linked to the
  principle via ``related``. The append-only store makes losing
  evidence structurally impossible; this engine also keeps the member
  concepts themselves.
- Candidate principles are groups of active concepts with *identical,
  non-empty prerequisite sets* — several ideas resting on exactly the
  same foundations are the classic shape of a hidden common principle.
  Richer similarity (shared relations, embedding distance) can extend
  ``propose`` later.
- Predictive performance is guarded by an injected
  :class:`PerformanceProbe`. If the probe's score drops by more than
  ``tolerance`` after applying, the principle is *retracted* (status
  change with reason — still never deleted). No probe means proposals
  are applied on structural evidence alone; the report says which.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.knowledge.graph import KnowledgeGraph
from allm.knowledge.types import Concept

logger = get_logger("compression.engine")


@runtime_checkable
class PerformanceProbe(Protocol):
    """Measures system performance (e.g. mean exam score) on demand."""

    def score(self) -> float: ...


class MergeProposal(BaseModel):
    """A candidate higher-level principle."""

    model_config = ConfigDict(frozen=True)

    principle: str
    members: tuple[str, ...]
    shared_prerequisites: tuple[str, ...]
    rationale: str


class CompressionOutcome(BaseModel):
    """What happened to one proposal."""

    model_config = ConfigDict(frozen=True)

    proposal: MergeProposal
    applied: bool
    retracted: bool = False
    score_before: float | None = None
    score_after: float | None = None
    detail: str


class CompressionEngine:
    """Proposes and applies evidence-preserving abstractions."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        probe: PerformanceProbe | None = None,
        *,
        min_group: int = 2,
        tolerance: float = 0.05,
    ) -> None:
        if min_group < 2:
            raise ValueError("a principle needs at least two member concepts")
        self._graph = graph
        self._probe = probe
        self._min_group = min_group
        self._tolerance = tolerance

    def propose(self) -> list[MergeProposal]:
        """Group active concepts by identical non-empty prerequisite sets."""
        groups: dict[tuple[str, ...], list[Concept]] = {}
        for concept in self._graph.concepts():
            if concept.status != "active" or not concept.prerequisites:
                continue
            key = tuple(sorted(concept.prerequisites))
            groups.setdefault(key, []).append(concept)
        proposals = []
        for prerequisites, members in sorted(groups.items()):
            if len(members) < self._min_group:
                continue
            names = tuple(sorted(c.name for c in members))
            principle = f"principle-of-{'+'.join(prerequisites)}"
            if self._graph.get(principle) is not None:
                continue  # already abstracted
            proposals.append(
                MergeProposal(
                    principle=principle,
                    members=names,
                    shared_prerequisites=prerequisites,
                    rationale=(
                        f"{len(names)} concepts ({', '.join(names)}) rest on the "
                        f"same foundations ({', '.join(prerequisites)})"
                    ),
                )
            )
        return proposals

    def apply(self, proposal: MergeProposal) -> CompressionOutcome:
        """Create the principle; retract it if the probe reports regression."""
        before = self._probe.score() if self._probe is not None else None
        members = [self._graph.get(name) for name in proposal.members]
        evidence = tuple(e for concept in members for e in concept.evidence)
        principle = Concept(
            name=proposal.principle,
            description=f"Common principle behind: {', '.join(proposal.members)}",
            prerequisites=proposal.shared_prerequisites,
            related=proposal.members,
            confidence=min(c.confidence for c in members),
            usefulness=max(c.usefulness for c in members),
            curiosity=max(c.curiosity for c in members),
            evidence=evidence,
            source="compression",
        )
        self._graph.add(principle, reason=proposal.rationale)
        for member in proposal.members:
            self._graph.revise(
                member,
                reason=f"abstracted under {proposal.principle}",
                add_related=[proposal.principle],
            )

        after = self._probe.score() if self._probe is not None else None
        if before is not None and after is not None and after < before - self._tolerance:
            self._graph.revise(
                proposal.principle,
                reason=(
                    f"retracted: performance dropped {before:.3f} -> {after:.3f} "
                    f"(tolerance {self._tolerance})"
                ),
                status="retracted",
            )
            logger.info("retracted %r after regression", proposal.principle)
            return CompressionOutcome(
                proposal=proposal,
                applied=True,
                retracted=True,
                score_before=before,
                score_after=after,
                detail="applied then retracted: predictive performance regressed",
            )
        logger.info("applied %r (%s)", proposal.principle, proposal.rationale)
        return CompressionOutcome(
            proposal=proposal,
            applied=True,
            score_before=before,
            score_after=after,
            detail="applied"
            + ("" if before is not None else " (no probe: structural evidence only)"),
        )

    def compress(self) -> list[CompressionOutcome]:
        """Propose and apply everything; returns one outcome per proposal."""
        return [self.apply(p) for p in self.propose()]
