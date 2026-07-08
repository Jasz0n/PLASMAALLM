"""Multimodal learning loop — Researcher + debate evidence (M9).

Dual plasma students disagree; unresolved debates trigger Researcher
visual evidence attached to episodic memory.

    PYTHONPATH=src python3 examples/36_multimodal_learning_loop.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.debate import DebateEngine
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp.corpus import DEFAULT_TOPIC, load_samples_jsonl
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


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-multimodal-loop-"))
    store = SQLiteRecordStore(workdir / "loop.sqlite3")
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC, description="Kids plasma"))

    samples = load_samples_jsonl(ROOT / "transcripts/Kids/samples_definitions.jsonl")[:24]
    holdout = [
        Sample(
            id="debate-q",
            input="Did the plasma field show instability when magnets chased each other?",
            target="unstable",
            metadata={"topic": DEFAULT_TOPIC},
        ),
    ]
    teacher = Teacher(
        KnowledgeState(store),
        DatasetExamGenerator(holdout),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )

    debate_prompt = holdout[0].input
    student_a = ModelStudent(
        "plasma-a",
        DEFAULT_TOPIC,
        EchoModel(ModelSpec(name="a", provider="echo", model_id="none")),
    )
    student_b = ModelStudent(
        "plasma-b",
        DEFAULT_TOPIC,
        EchoModel(ModelSpec(name="b", provider="echo", model_id="none")),
    )
    trainer = InContextTrainer()
    trainer.train(student_a, [Sample(id="a1", input=debate_prompt, target="yes unstable", metadata={"topic": DEFAULT_TOPIC})])
    trainer.train(student_b, [Sample(id="b1", input=debate_prompt, target="no stable", metadata={"topic": DEFAULT_TOPIC})])
    trainer.train(student_a, samples[:8])
    trainer.train(student_b, samples[:8])

    pool = SamplePool()
    pool.ingest(samples)

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
        graph=graph,
        state=teacher.state,
        student_ids=("plasma-a", "plasma-b"),
    )
    research_report = researcher.run_cycle()

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
            iterations=2,
            questions_per_exam=3,
            samples_per_iteration=8,
            study_failures=False,
            seed=11,
            enable_debate_evidence=True,
        ),
    )

    print("\n=== M9: Multimodal learning loop ===")
    print(f"  researcher packages: {len(research_report.packages)}")
    print(f"  multimodal synced: {len(research_report.multimodal_synced)}")

    reports = loop.run()
    for report in reports:
        for row in report.students:
            print(
                f"  iter {report.iteration} {row.student_id}: "
                f"{row.score_before:.2f}->{row.score_after:.2f} studied={row.samples_studied}"
            )
        if report.debate_disagreement is not None:
            print(f"    debate disagreement: {report.debate_disagreement:.2f}")
        evidence = report.debate_evidence
        if isinstance(evidence, DebateEvidenceSummary):
            print(f"    debate evidence found: {evidence.found} hits={evidence.hit_count}")
            if evidence.found:
                print(f"    evidence: {evidence.summary[:120]}...")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
