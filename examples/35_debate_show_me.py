"""Debate with multimodal 'show me' evidence (M8).

    PYTHONPATH=src python3 examples/35_debate_show_me.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.debate import DebateEngine, resolve_debate_evidence
from allm.exam.base import Question
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.evidence_broker import EvidenceBroker
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import teacher_show_me

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-debate-show-me-"))
    store = SQLiteRecordStore(workdir / "debate.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()
    broker = EvidenceBroker.from_store(store)

    print("\n=== M8: Researcher multimodal + debate evidence ===")
    unsynced = [s for s in report.curiosity_signals if s.source == "unsynced_video"]
    if unsynced:
        print(f"  curiosity (unsynced video): {unsynced[0].question} score={unsynced[0].score:.2f}")

    question = Question(
        id="plasma-stability",
        prompt="Did the plasma field show instability when the magnets chased each other?",
        expected=None,
        topic=DEFAULT_TOPIC,
        kind="factual",
    )
    student_a = ScriptedStudent("plasma-a", DEFAULT_TOPIC, knowledge={question.prompt: "yes unstable"})
    student_b = ScriptedStudent("plasma-b", DEFAULT_TOPIC, knowledge={question.prompt: "no it was stable"})

    engine = DebateEngine(disagreement_threshold=0.3)
    debate = engine.debate(question, [student_a, student_b])

    print(f"\n=== Debate ===")
    print(f"  disagreement: {debate.disagreement:.2f}")
    print(f"  unresolved: {debate.unresolved}")
    for cluster in debate.clusters:
        print(f"    cluster: {cluster.answer_text!r} ({cluster.size} students)")

    print(f"\n=== Student B: 'Show me' ===")
    show_me = teacher_show_me(
        broker,
        asker_id="plasma-b",
        topic=DEFAULT_TOPIC,
        query="blue plasma magnet",
    )
    print(f"  found: {show_me.found}")
    print(f"  teacher: {show_me.teacher_note}")
    for hit in show_me.bundle.hits[:2]:
        row = hit.evidence
        frames = ""
        if row.visual and row.visual.frame_start is not None:
            frames = f" frames {row.visual.frame_start}-{row.visual.frame_end}"
        print(f"    {row.source_id} @{row.timestamp_sec:.0f}s{frames} conf={row.confidence:.2f}")

    resolution = resolve_debate_evidence(broker, debate, query="plasma magnet")
    print(f"\n=== Debate evidence resolution ===")
    print(f"  query: {resolution.query}")
    print(f"  summary: {resolution.bundle.summary}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
