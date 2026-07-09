"""The live workshop loop (M51): observation → synced evidence → KDP.

A LiveKit workshop stream, folded into the knowledge graph tick by tick
and announced on the M51 event feed — the same evidence machinery that
serves documents and practice, now fed by live observation.

Offline (default) drives the loop with canned synced evidence so it is
deterministic and dependency-free. Against a real stream you would bind
a LiveKit observer instead::

    from allm.researcher import observer_source
    from allm.researcher.livekit_observer import LiveKitRtcObserver
    source = observer_source(LiveKitRtcObserver(), stream, creds, cache_dir=...)
    loop = WorkshopLoop(store, source, topic="plasma", events=log, board=board)

    PYTHONPATH=src python3 examples/80_workshop_loop.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.events import EventLog
from allm.evidence import EvidenceBinder, EvidenceLedger
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard
from allm.researcher import WorkshopLoop
from allm.researcher.multimodal_types import AudioCue, SyncedEvidence, VisualCue
from allm.storage import SQLiteRecordStore

# Three "moments" of a live plasma workshop, as an observer would sync them.
STREAM = "workshop-plasma-live"
TICKS = [
    [
        SyncedEvidence(
            source_id=STREAM, timestamp_sec=6, live_stream_id=STREAM, is_live=True,
            transcript_excerpt="A plasma is an ionized gas; we call it the fourth state of matter.",
            visual=VisualCue(timestamp_sec=6, description="A glowing plasma ball with visible filaments."),
        ),
        SyncedEvidence(
            source_id=STREAM, timestamp_sec=14, live_stream_id=STREAM, is_live=True,
            transcript_excerpt="Plasma conducts electricity because its electrons are free.",
            audio=AudioCue(timestamp_sec=14, description="Instructor explains conduction."),
        ),
    ],
    [
        SyncedEvidence(
            source_id=STREAM, timestamp_sec=25, live_stream_id=STREAM, is_live=True,
            transcript_excerpt="Magnetic confinement holds a fusion plasma away from the vessel walls.",
            visual=VisualCue(timestamp_sec=25, description="A tokamak cross-section diagram.", is_diagram=True),
        ),
    ],
]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-workshop-"))
    store = SQLiteRecordStore(workdir / "workshop.sqlite3")
    log = EventLog(store)
    graph = KnowledgeGraph(store)
    board = ProposalBoard(store, EvidenceBinder(graph, EvidenceLedger(store)))

    scripted = iter(TICKS)

    def source() -> list[SyncedEvidence]:
        return next(scripted, [])

    loop = WorkshopLoop(store, source, topic="plasma", events=log, board=board)
    report = loop.run(len(TICKS))

    print("\n=== M51: live workshop loop ===")
    for tick in report.ticks:
        print(
            f"  tick {tick.tick}: {tick.rows_observed} rows observed, "
            f"{tick.documents_ingested} distilled → {tick.concepts_total} concepts, "
            f"{len(tick.conflicts_opened)} conflict(s)"
        )
    print(f"\n  totals: {report.rows_observed} rows → {report.concepts_total} concepts")

    print("\n  live feed:")
    for event in log.since(0):
        print(f"    #{event.seq} {event.type} {event.subject} {event.data}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
