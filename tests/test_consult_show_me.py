"""Tests for consultation show me integration."""

from pathlib import Path

from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.students import ScriptedStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, mediated_consultation
from allm.teacher.show_me import ConsultationEvidence, derive_show_me_query

ROOT = Path(__file__).resolve().parents[1]


def test_derive_show_me_query_from_prompt() -> None:
    query = derive_show_me_query("Did the plasma become unstable in the video?", DEFAULT_TOPIC)
    assert query == "plasma"


def test_mediated_consultation_show_me_on_reject() -> None:
    store = SQLiteRecordStore(":memory:")
    state = KnowledgeState(store)
    prompt = "Did the plasma show magnet motion in the video?"
    expected = "yes magnetical beat"
    topic = DEFAULT_TOPIC
    sample = Sample(id="s1", input=prompt, target=expected, metadata={"topic": topic})
    teacher = Teacher(state, DatasetExamGenerator([sample]), ExactMatchGrader("contains"))
    teacher.evaluate(
        ScriptedStudent("plasma-expert", topic, knowledge={prompt: expected}),
        teacher.create_exam(num_questions=1, seed=1),
    )

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=2,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    researcher.run_cycle()
    broker = researcher.evidence_broker()

    wrong_expert = ScriptedStudent(
        "plasma-expert",
        topic,
        knowledge={prompt: "no stable"},
    )
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
    assert not result.approved
    assert result.show_me_requested
    assert isinstance(result.evidence, ConsultationEvidence)
    assert result.evidence.found
    assert result.evidence.hit_count > 0
