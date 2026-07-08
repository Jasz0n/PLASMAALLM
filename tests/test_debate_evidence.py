"""Debate + multimodal evidence integration tests."""

from pathlib import Path

from allm.data.base import Sample
from allm.debate import DebateEngine, resolve_debate_evidence
from allm.exam import ExactMatchGrader
from allm.exam.base import Question
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.evidence_broker import EvidenceBroker
from allm.researcher.multimodal import unsynced_video_gap
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import teacher_show_me

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP_DIR = ROOT / "transcripts/Kids/cleaned/mk"
VISUAL_DIR = ROOT / "transcripts/Kids/visual"


def test_unsynced_video_gap_detects_missing_cues() -> None:
    mentions, gap = unsynced_video_gap(WORKSHOP_DIR, VISUAL_DIR)
    assert mentions > 0
    assert gap >= 0


def test_evidence_broker_show_me() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        workshop_dir=WORKSHOP_DIR,
        video_fixture_dir=VISUAL_DIR,
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    researcher.run_cycle()
    broker = EvidenceBroker.from_store(store)
    bundle = broker.show_me("blue plasma", topic=DEFAULT_TOPIC)
    assert bundle.hits
    assert bundle.confidence > 0
    assert "frames" in bundle.summary


def test_resolve_debate_evidence() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        workshop_dir=WORKSHOP_DIR,
        video_fixture_dir=VISUAL_DIR,
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    researcher.run_cycle()
    broker = EvidenceBroker.from_store(store)

    question = Question(
        id="debate-1",
        prompt="Did the plasma become unstable during the magnet demonstration?",
        expected=None,
        topic=DEFAULT_TOPIC,
        kind="factual",
    )
    students = [
        ScriptedStudent("a", DEFAULT_TOPIC, knowledge={question.prompt: "yes unstable"}),
        ScriptedStudent("b", DEFAULT_TOPIC, knowledge={question.prompt: "no it stayed stable"}),
    ]
    result = DebateEngine(disagreement_threshold=0.3).debate(question, students)
    resolution = resolve_debate_evidence(broker, result, query="plasma")
    assert resolution.unresolved
    assert resolution.bundle.hits or resolution.bundle.summary


def test_teacher_show_me() -> None:
    store = SQLiteRecordStore(":memory:")
    researcher = ResearcherLayer(
        store,
        workshop_dir=WORKSHOP_DIR,
        video_fixture_dir=VISUAL_DIR,
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    researcher.run_cycle()
    broker = EvidenceBroker.from_store(store)
    result = teacher_show_me(
        broker,
        asker_id="software-student",
        topic=DEFAULT_TOPIC,
        query="magnet rotation",
    )
    assert result.found
    assert result.teacher_note
