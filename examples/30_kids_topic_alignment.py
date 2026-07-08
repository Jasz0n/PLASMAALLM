"""Kids plasma: Researcher workshop concepts aligned to pool topic (offline).

Demonstrates that KDP fragment labels map to ``kids-plasma`` so planner
goals match the sample pool and ``studied > 0``.

    PYTHONPATH=src python3 examples/30_kids_topic_alignment.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.kdp.curriculum import load_curriculum_splits
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import KelSteeredLearningLoop, KelSteeringConfig, LoopConfig
from allm.memory import EpisodicMemory
from allm.models import EchoModel, ModelSpec
from allm.planner import NeedPlanner
from allm.researcher import ResearcherLayer
from allm.students import FailureLog, ModelStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import InContextTrainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    train, holdout = load_curriculum_splits(ROOT)
    if len(train) < 8:
        raise SystemExit("Need Kids curriculum splits")

    workdir = Path(tempfile.mkdtemp(prefix="allm-kids-align-"))
    store = SQLiteRecordStore(workdir / "loop.sqlite3")
    state = KnowledgeState(store)
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC, description="Kids plasma science"))
    kel = KnowledgeEvaluationLayer(graph, store, state)

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        workshop_max_files=2,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()
    rec_topics = {rec.topic for rec in report.recommendations}
    print("\n=== Topic alignment (workshop → curriculum) ===")
    print(f"  package concepts: {len(report.packages[0].concepts) if report.packages else 0}")
    print(f"  recommendation topics: {sorted(rec_topics)}")
    print(f"  aligned to pool topic: {DEFAULT_TOPIC in rec_topics}")

    teacher = Teacher(
        state,
        DatasetExamGenerator(holdout[: min(4, len(holdout))]),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )
    pool = SamplePool()
    pool.ingest(train[: min(64, len(train))])
    student = ModelStudent(
        "kids-aligned",
        DEFAULT_TOPIC,
        EchoModel(ModelSpec(name="echo", provider="echo", model_id="none")),
    )
    trainer = InContextTrainer()
    trainer.train(student, train[: min(32, len(train))])

    loop = KelSteeredLearningLoop(
        kel=kel,
        steering=KelSteeringConfig(
            min_iterations_before_halt=2,
            min_lg_history_for_halt=2,
            halt_on_static_illusion=False,
        ),
        researcher=researcher,
        teacher=teacher,
        students=[student],
        planner=NeedPlanner(),
        trainer=trainer,
        pool=pool,
        memory=EpisodicMemory(store),
        failure_log=FailureLog(store),
        graph=graph,
        config=LoopConfig(
            iterations=2,
            questions_per_exam=min(4, len(holdout)),
            samples_per_iteration=min(16, len(train)),
            study_failures=False,
            strategy="definitions",
            sample_kinds=("definition", "we_call", "compact"),
            seed=11,
        ),
    )

    print("\n=== KEL loop (aligned goals should study pool samples) ===")
    reports = loop.run()
    for report in reports:
        row = report.students[0]
        print(
            f"  iter {report.iteration}: goals={row.goals[:3]} studied={row.samples_studied}"
        )

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
