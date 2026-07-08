"""Learning loop debate evidence integration."""

from pathlib import Path

from allm.collector import SamplePool
from allm.data.base import Sample
from allm.debate import DebateEngine
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import LearningLoop, LoopConfig
from allm.loop.debate_evidence import DebateEvidenceSummary
from allm.memory import EpisodicMemory
from allm.models import EchoModel, ModelSpec
from allm.planner import NeedPlanner
from allm.researcher import ResearcherLayer
from allm.students import FailureLog, ModelStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import InContextTrainer

ROOT = Path(__file__).resolve().parents[1]


def test_loop_attaches_debate_evidence_when_unresolved() -> None:
    store = SQLiteRecordStore(":memory:")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC))

    prompt = "Did plasma become unstable in the magnet demo?"
    holdout = [
        Sample(id="q1", input=prompt, target="yes", metadata={"topic": DEFAULT_TOPIC}),
    ]
    teacher = Teacher(
        KnowledgeState(store),
        DatasetExamGenerator(holdout),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )

    student_a = ModelStudent("a", DEFAULT_TOPIC, EchoModel(ModelSpec(name="a", provider="echo", model_id="none")))
    student_b = ModelStudent("b", DEFAULT_TOPIC, EchoModel(ModelSpec(name="b", provider="echo", model_id="none")))
    trainer = InContextTrainer()
    trainer.train(student_a, [Sample(id="s1", input=prompt, target="yes", metadata={"topic": DEFAULT_TOPIC})])
    trainer.train(student_b, [Sample(id="s2", input=prompt, target="no", metadata={"topic": DEFAULT_TOPIC})])

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
        graph=graph,
        state=teacher.state,
    )
    researcher.run_cycle()

    pool = SamplePool()
    pool.ingest([Sample(id="t1", input="plasma?", target="state", metadata={"topic": DEFAULT_TOPIC})])

    loop = LearningLoop(
        teacher=teacher,
        students=[student_a, student_b],
        planner=NeedPlanner(),
        trainer=trainer,
        pool=pool,
        memory=EpisodicMemory(store),
        failure_log=FailureLog(store),
        graph=graph,
        debate=DebateEngine(disagreement_threshold=0.3),
        researcher=researcher,
        config=LoopConfig(
            iterations=1,
            questions_per_exam=1,
            samples_per_iteration=1,
            study_failures=False,
            seed=3,
            enable_debate_evidence=True,
        ),
    )
    reports = loop.run()
    assert reports[0].debate_disagreement is not None
    evidence = reports[0].debate_evidence
    assert isinstance(evidence, DebateEvidenceSummary)
    assert evidence.hit_count >= 0
