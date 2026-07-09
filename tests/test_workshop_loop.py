"""The live workshop loop (M51): observation → evidence → KDP, offline."""

from pathlib import Path

import pytest

from allm.events import EventLog
from allm.evidence import EvidenceBinder, EvidenceLedger
from allm.knowledge import KnowledgeGraph
from allm.proposals import ProposalBoard
from allm.researcher.multimodal_types import AudioCue, SyncedEvidence, VisualCue
from allm.researcher.workshop_loop import WorkshopLoop, _row_text, observer_source
from allm.storage import SQLiteRecordStore


def _live_rows() -> list[SyncedEvidence]:
    return [
        SyncedEvidence(
            source_id="live-plasma", timestamp_sec=5,
            transcript_excerpt="A plasma is an ionized gas; we call it the fourth state of matter.",
            visual=VisualCue(timestamp_sec=5, description="A glowing plasma ball with filaments."),
            live_stream_id="stream-42", is_live=True,
        ),
        SyncedEvidence(
            source_id="live-plasma", timestamp_sec=12,
            transcript_excerpt="Plasma conducts electricity because its electrons are free.",
            audio=AudioCue(timestamp_sec=12, description="Instructor explains conduction."),
            live_stream_id="stream-42", is_live=True,
        ),
    ]


def test_folds_live_evidence_into_the_graph_and_announces_it(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "ws.sqlite3")
    log = EventLog(store)
    loop = WorkshopLoop(store, _live_rows, topic="plasma", events=log)

    tick = loop.tick()
    assert tick.rows_observed == 2 and tick.documents_ingested == 2
    assert tick.stream_id == "stream-42"
    assert tick.concepts_total > 0  # KDP distilled the observation into concepts

    events = log.since(0)
    assert [e.type for e in events] == ["workshop.observed"]
    assert events[0].subject == "stream-42"
    assert events[0].data["is_live"] is True and events[0].data["rows"] == 2


def test_report_rolls_up_multiple_ticks(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "ws.sqlite3")
    loop = WorkshopLoop(store, _live_rows, topic="plasma")  # no events/board
    report = loop.run(3)
    assert len(report.ticks) == 3
    assert report.rows_observed == 6
    assert report.concepts_total == report.ticks[-1].concepts_total > 0
    assert [t.tick for t in report.ticks] == [1, 2, 3]


def test_empty_observation_is_a_quiet_noop(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "ws.sqlite3")
    log = EventLog(store)
    loop = WorkshopLoop(store, lambda: [], events=log)
    tick = loop.tick()
    assert tick.rows_observed == 0 and tick.event_seq is None
    assert log.since(0) == []  # nothing observed, nothing announced


def test_conflicts_open_proposals_on_the_feed(tmp_path: Path) -> None:
    store = SQLiteRecordStore(tmp_path / "ws.sqlite3")
    log = EventLog(store)
    graph = KnowledgeGraph(store)
    board = ProposalBoard(store, EvidenceBinder(graph, EvidenceLedger(store)))
    loop = WorkshopLoop(store, _live_rows, topic="plasma", events=log, board=board)

    tick = loop.tick()
    # board is wired without error; any opened proposal is announced and tracked
    assert isinstance(tick.conflicts_opened, tuple)
    opened = {e.subject for e in log.since(0) if e.type == "proposal.opened"}
    assert opened == set(tick.conflicts_opened)


def test_row_text_bridges_transcript_visual_and_audio() -> None:
    row = SyncedEvidence(
        source_id="s", timestamp_sec=1, transcript_excerpt="the arc glows",
        visual=VisualCue(timestamp_sec=1, description="bright arc", ocr_text="10 kV"),
        audio=AudioCue(timestamp_sec=1, description="a hum"),
    )
    text = _row_text(row)
    assert "the arc glows" in text and "bright arc" in text
    assert "10 kV" in text and "a hum" in text
    # a row with no usable text yields nothing to ingest
    assert _row_text(SyncedEvidence(source_id="s", timestamp_sec=1, transcript_excerpt="")) == ""


def test_observer_source_adapts_a_real_observer() -> None:
    captured = {}

    class FakeObserver:
        def observe(self, stream, credentials, *, cache_dir, capture_seconds):
            captured["args"] = (stream, credentials, cache_dir, capture_seconds)
            return _live_rows()

    source = observer_source(
        FakeObserver(), "stream-obj", "creds", cache_dir="/tmp/x", capture_seconds=2.0
    )
    rows = source()
    assert len(rows) == 2
    assert captured["args"] == ("stream-obj", "creds", "/tmp/x", 2.0)
