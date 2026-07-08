"""Mediated consultation with Teacher 'show me' evidence (M11).

    PYTHONPATH=src python3 examples/39_consult_show_me.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.evidence_broker import EvidenceBroker
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, mediated_consultation
from allm.teacher.show_me import ConsultationEvidence

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-consult-show-me-"))
    store = SQLiteRecordStore(workdir / "consult.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    researcher.run_cycle()
    broker = EvidenceBroker.from_store(store)

    prompt = "Did the plasma become unstable when magnets chased each other in the video?"
    expected = "the field showed magnetical beat and rotation"
    topic = DEFAULT_TOPIC

    state = KnowledgeState(store)
    sample = Sample(id="s1", input=prompt, target=expected, metadata={"topic": topic})
    teacher = Teacher(
        state,
        DatasetExamGenerator([sample]),
        ExactMatchGrader("contains"),
    )
    teacher.evaluate(
        ScriptedStudent("plasma-expert", topic, knowledge={prompt: expected}),
        teacher.create_exam(num_questions=1, seed=1),
    )

    wrong_expert = ScriptedStudent(
        "plasma-expert",
        topic,
        knowledge={prompt: "no it was completely stable with no motion"},
    )

    print("\n=== M11: Mediated consultation + show me ===")
    result = mediated_consultation(
        state,
        ExactMatchGrader("contains"),
        "software-student",
        wrong_expert,
        topic=topic,
        prompt=prompt,
        expected=expected,
        evidence_broker=broker,
        show_me_on_reject=True,
    )

    print(f"  asker: software-student")
    print(f"  expert: {result.expert_id}")
    print(f"  approved: {result.approved}")
    print(f"  show me requested: {result.show_me_requested}")
    print(f"  reason: {result.reason}")

    evidence = result.evidence
    if isinstance(evidence, ConsultationEvidence):
        print(f"\n=== Teacher show me ===")
        print(f"  query: {evidence.query}")
        print(f"  found: {evidence.found}")
        print(f"  hits: {evidence.hit_count}")
        print(f"  confidence: {evidence.confidence:.2f}")
        print(f"  summary: {evidence.summary[:160]}...")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
