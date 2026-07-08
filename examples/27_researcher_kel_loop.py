"""Researcher → Teacher (KEL) → Student loop (offline).

Runs a Researcher discovery cycle, merges recommendations into the
planner catalog, then executes a short KEL-steered learning loop with
an echo student (no GPU). Uses the software fixture so package concepts
match sample topics in the pool.

    PYTHONPATH=src python3 examples/27_researcher_kel_loop.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
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
SOFTWARE_SAMPLES = ROOT / "transcripts/Software/samples_dev.jsonl"


def load_software_samples(path: Path) -> list[Sample]:
    rows: list[Sample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            Sample(
                id=row["id"],
                input=row["input"],
                target=row.get("target"),
                metadata={
                    "topic": row.get("topic", "software"),
                    "sample_kind": row.get("sample_kind", "definition"),
                },
            )
        )
    return rows


def main() -> None:
    setup_logging("INFO")
    samples = load_software_samples(SOFTWARE_SAMPLES)
    holdout = samples[-4:]
    train = samples[:-4]

    workdir = Path(tempfile.mkdtemp(prefix="allm-researcher-kel-"))
    store = SQLiteRecordStore(workdir / "loop.sqlite3")
    state = KnowledgeState(store)
    graph = KnowledgeGraph(store)
    for topic in ("fastify-api", "prisma-orm", "typescript-react"):
        graph.add(Concept(name=topic, description="Software development"))
    kel = KnowledgeEvaluationLayer(graph, store, state)

    print("\n=== Researcher discovery (pre-loop) ===")
    researcher = ResearcherLayer(
        store,
        software_samples=SOFTWARE_SAMPLES,
        workshop_max_files=0,
    )
    report = researcher.run_cycle()
    print(f"  packages: {len(report.packages)}  recommendations: {len(report.recommendations)}")

    teacher = Teacher(
        state,
        DatasetExamGenerator(holdout),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )
    pool = SamplePool()
    pool.ingest(train)
    student = ModelStudent(
        "software-researcher",
        "fastify-api",
        EchoModel(ModelSpec(name="echo", provider="echo", model_id="none")),
    )
    trainer = InContextTrainer()
    trainer.train(student, train)

    loop = KelSteeredLearningLoop(
        kel=kel,
        steering=KelSteeringConfig(
            min_iterations_before_halt=3,
            min_lg_history_for_halt=3,
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
            iterations=3,
            questions_per_exam=len(holdout),
            samples_per_iteration=len(train),
            study_failures=False,
            strategy="definitions",
            sample_kinds=("definition", "teaching", "compact"),
            seed=7,
        ),
    )

    print("\n=== KEL-steered loop (Researcher-boosted catalog) ===")
    reports = loop.run()
    for report in reports:
        row = report.students[0]
        print(
            f"  iter {report.iteration}: {row.score_before:.2f} -> {row.score_after:.2f} "
            f"goals={row.goals[:3]} studied={row.samples_studied}"
        )

    kel_report = kel.evaluate()
    lg = kel_report.lg if kel_report.lg is not None else 0.0
    ghs = kel_report.ghs if kel_report.ghs is not None else 0.0
    print(f"\n  KEL learning gain: {lg:.3f}  graph health: {ghs:.3f}")
    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
