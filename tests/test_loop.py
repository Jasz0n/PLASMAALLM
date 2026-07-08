"""Integration test: the full continuous learning loop, offline."""

from pathlib import Path

import pytest

from allm.collector import SamplePool
from allm.data.base import Sample
from allm.debate import DebateEngine
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import LearningLoop, LoopConfig
from allm.memory import EpisodicMemory
from allm.models import EchoModel, ModelSpec
from allm.planner import NeedPlanner
from allm.students import FailureLog, ModelStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.tracking import LocalTracker
from allm.trainer import InContextTrainer

FACTS = {
    "2+2?": ("4", "math"),
    "3*3?": ("9", "math"),
    "Capital of France?": ("Paris", "geography"),
    "Capital of Japan?": ("Tokyo", "geography"),
}


def make_samples() -> list[Sample]:
    return [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, (a, topic)) in enumerate(FACTS.items())
    ]


def make_student(sid: str) -> ModelStudent:
    model = EchoModel(ModelSpec(name=sid, provider="echo", model_id="none"))
    return ModelStudent(sid, "general", model)


@pytest.fixture()
def loop(tmp_path: Path):
    store = SQLiteRecordStore(tmp_path / "loop.sqlite3")
    samples = make_samples()
    teacher = Teacher(
        KnowledgeState(store),
        DatasetExamGenerator(samples),
        ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=1.0),
    )
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="basics"))
    graph.add(Concept(name="math", prerequisites=("basics",), usefulness=0.9))
    graph.add(Concept(name="geography", prerequisites=("basics",), usefulness=0.6))
    pool = SamplePool()
    pool.ingest(samples)
    tracker = LocalTracker(tmp_path / "runs")
    run = tracker.start_run("loop-test")
    instance = LearningLoop(
        teacher=teacher,
        students=[make_student("alpha"), make_student("beta")],
        planner=NeedPlanner(),
        trainer=InContextTrainer(),
        pool=pool,
        memory=EpisodicMemory(store),
        failure_log=FailureLog(store),
        graph=graph,
        compression=None,
        debate=DebateEngine(grader=ExactMatchGrader()),
        run=run,
        config=LoopConfig(iterations=2, questions_per_exam=4, seed=7),
    )
    yield instance, teacher, run, tmp_path
    store.close()


def test_loop_produces_learning(loop) -> None:
    instance, teacher, run, tmp_path = loop
    reports = instance.run()
    assert len(reports) == 2

    first = reports[0]
    # students start clueless (echo model) and learn from failures
    for student_report in first.students:
        assert student_report.score_before == 0.0
        assert student_report.score_after > student_report.score_before
        assert student_report.goals  # planner assigned something
        assert student_report.samples_studied > 0

    # by iteration 2 the failure-driven notes cover the whole pool
    for student_report in reports[1].students:
        assert student_report.score_after == 1.0

    # debate ran between the two students
    assert first.debate_disagreement is not None

    # everything was persisted: exams, goals, run metrics
    assert teacher.state.exam_results("alpha")
    assert teacher.state.current_goals("alpha")
    metrics_file = run.directory / "metrics.jsonl"
    assert metrics_file.exists() and metrics_file.read_text().strip()


def test_loop_requires_students(loop) -> None:
    instance, teacher, run, tmp_path = loop
    with pytest.raises(ValueError, match="at least one student"):
        LearningLoop(
            teacher=teacher,
            students=[],
            planner=NeedPlanner(),
            trainer=InContextTrainer(),
            pool=SamplePool(),
            memory=instance._memory,  # reuse fixtures; construction fails first
            failure_log=instance._failures,
        )
