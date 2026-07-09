"""The live workshop loop (Roadmap M51): observation → evidence → KDP.

The LiveKit stack already turns a room into :class:`SyncedEvidence`
(fixture offline, RTC when livekit is installed). This is the missing
orchestration: a repeatable loop that folds each batch of live evidence
into the knowledge graph through KDP and announces it on the M51 event
stream, so live workshops feed the same evidence machinery as documents
and practice — and show up on the dashboard and any approved webhook.

Decoupled from LiveKit credentials on purpose: the loop pulls from a
``source`` callable that yields the next batch of evidence. In
production that callable wraps a real observer (:func:`observer_source`);
in tests it is a canned list. Same loop, real or fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel, ConfigDict

from allm.core.logging import get_logger
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.knowledge import KnowledgeGraph
from allm.researcher.multimodal_types import SyncedEvidence
from allm.storage.base import RecordStore

if TYPE_CHECKING:  # optional collaborators — the loop works without them
    from allm.events import EventLog
    from allm.proposals import ProposalBoard

logger = get_logger("researcher.workshop_loop")

EvidenceSource = Callable[[], list[SyncedEvidence]]


class WorkshopTick(BaseModel):
    """What one observation cycle folded into the graph."""

    model_config = ConfigDict(frozen=True)

    tick: int
    stream_id: str
    rows_observed: int  # synced-evidence rows this tick
    documents_ingested: int  # rows that carried usable text
    concepts_total: int  # graph size after this tick
    conflicts_opened: tuple[str, ...] = ()  # proposal ids
    event_seq: int | None = None  # the workshop.observed event, if emitted


class WorkshopReport(BaseModel):
    """The whole loop: one entry per tick plus roll-ups."""

    model_config = ConfigDict(frozen=True)

    ticks: tuple[WorkshopTick, ...]

    @property
    def rows_observed(self) -> int:
        return sum(t.rows_observed for t in self.ticks)

    @property
    def concepts_total(self) -> int:
        return self.ticks[-1].concepts_total if self.ticks else 0

    @property
    def conflicts_opened(self) -> tuple[str, ...]:
        return tuple(pid for t in self.ticks for pid in t.conflicts_opened)


class WorkshopLoop:
    """Fold live observation into the graph, tick by tick."""

    def __init__(
        self,
        store: RecordStore,
        source: EvidenceSource,
        *,
        topic: str = "workshop",
        events: "EventLog | None" = None,
        board: "ProposalBoard | None" = None,
    ) -> None:
        self._store = store
        self._source = source
        self._topic = topic
        self._events = events
        self._board = board
        self._documents = DocumentStore(store)
        self._graph = KnowledgeGraph(store)
        self._tick = 0

    def tick(self) -> WorkshopTick:
        """Observe once, distill the accumulated stream, fold in conflicts."""
        self._tick += 1
        rows = list(self._source())
        stream_id = _stream_id(rows)

        ingested = 0
        for row in rows:
            text = _row_text(row)
            if not text:
                continue
            self._documents.ingest_text(
                f"{row.source_id}@{row.timestamp_sec:.0f}s", text, context=self._topic
            )
            ingested += 1

        result = KDPipeline().distill(self._documents)
        GraphInjector(self._graph, self._store).inject(result)

        conflicts: list[str] = []
        if self._board is not None:
            for conflict in result.conflicts:
                proposal = self._board.from_conflict(conflict)
                conflicts.append(proposal.id)
                if self._events is not None:
                    self._events.emit(
                        "proposal.opened",
                        proposal.id,
                        {"question": proposal.question, "origin": "workshop"},
                    )

        concepts_total = len(self._graph.concepts())
        event_seq = None
        if self._events is not None and rows:
            event = self._events.emit(
                "workshop.observed",
                stream_id,
                {
                    "tick": self._tick,
                    "rows": len(rows),
                    "concepts_total": concepts_total,
                    "conflicts_opened": len(conflicts),
                    "is_live": any(r.is_live for r in rows),
                },
            )
            event_seq = event.seq

        logger.info(
            "workshop tick %d: %d rows, %d ingested, %d concepts, %d conflicts",
            self._tick, len(rows), ingested, concepts_total, len(conflicts),
        )
        return WorkshopTick(
            tick=self._tick,
            stream_id=stream_id,
            rows_observed=len(rows),
            documents_ingested=ingested,
            concepts_total=concepts_total,
            conflicts_opened=tuple(conflicts),
            event_seq=event_seq,
        )

    def run(self, ticks: int) -> WorkshopReport:
        """Run ``ticks`` observation cycles and return the full report."""
        return WorkshopReport(ticks=tuple(self.tick() for _ in range(ticks)))


def _stream_id(rows: list[SyncedEvidence]) -> str:
    for row in rows:
        if row.live_stream_id:
            return row.live_stream_id
    return rows[0].source_id if rows else "workshop"


def _row_text(row: SyncedEvidence) -> str:
    """The teachable text in one synced-evidence row, for KDP."""
    parts = [row.transcript_excerpt]
    if row.visual is not None:
        parts.append(row.visual.description)
        if row.visual.ocr_text:
            parts.append(row.visual.ocr_text)
    if row.audio is not None:
        parts.append(row.audio.description)
    return " ".join(p.strip() for p in parts if p and p.strip())


def observer_source(
    observer,
    stream,
    credentials,
    *,
    cache_dir,
    capture_seconds: float = 3.0,
) -> EvidenceSource:
    """Bind a real :class:`LiveKitObserver` into a loop ``source``.

    Each call observes the room afresh — point the loop at this to run
    against a live stream instead of a fixture list.
    """

    def source() -> list[SyncedEvidence]:
        return observer.observe(
            stream, credentials, cache_dir=cache_dir, capture_seconds=capture_seconds
        )

    return source
