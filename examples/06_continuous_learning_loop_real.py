"""M1 demo: continuous learning loop with real Ollama models.

Runs the same cycle as ``06_continuous_learning_loop.py`` but students
answer via a small local model and grading uses a stronger judge when
exact-match is insufficient.

Requires a running Ollama daemon::

    ollama serve
    ollama pull qwen2.5:0.5b-instruct   # student (or set ALLM_STUDENT_MODEL)
    ollama pull qwen2.5:14b-instruct    # local judge (optional)

Environment:
    ALLM_STUDENT_MODEL   — Ollama tag for students (default: qwen2.5:0.5b-instruct)
    ALLM_GRADER          — ``local`` (default) or ``cloud`` for the judge model
    ALLM_ITERATIONS      — loop iterations (default: 5, M1 exit criterion)
    ALLM_EXAM            — ``dataset`` (default) or ``generative`` for ModelExamGenerator
    ALLM_WRITER_MODEL    — Ollama tag for generative exams (default: qwen2.5:7b-instruct)

    python examples/06_continuous_learning_loop_real.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.compression import CompressionEngine
from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.debate import DebateEngine
from allm.evaluation import evaluate_student
from allm.evaluation.calibration import calibration_comparison
from allm.exam import (
    DatasetExamGenerator,
    EscalatingGrader,
    ExactMatchGrader,
    LLMJudgeGrader,
    ModelExamGenerator,
)
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import LearningLoop, LoopConfig
from allm.memory import EpisodicMemory
from allm.models import ModelSpec, load_model
from allm.planner import NeedPlanner
from allm.students import FailureLog, ModelStudent
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.tracking import LocalTracker
from allm.trainer import InContextTrainer

ROOT = Path(__file__).resolve().parents[1]
FACTS = {
    "2+2?": ("4", "math"),
    "3*3?": ("9", "math"),
    "10/2?": ("5", "math"),
    "Capital of France?": ("Paris", "geography"),
    "Capital of Japan?": ("Tokyo", "geography"),
    "Capital of Egypt?": ("Cairo", "geography"),
}


def student_spec() -> ModelSpec:
    model_id = os.environ.get("ALLM_STUDENT_MODEL", "qwen2.5:0.5b-instruct")
    path = ROOT / "configs/models/ollama_student.yaml"
    spec = ModelSpec.from_yaml(path)
    return spec.model_copy(update={"model_id": model_id})


def grader_spec() -> ModelSpec:
    use_cloud = os.environ.get("ALLM_GRADER", "local").lower() == "cloud"
    name = "ollama_grader_cloud.yaml" if use_cloud else "ollama_grader_local.yaml"
    return ModelSpec.from_yaml(ROOT / "configs/models" / name)


def writer_spec() -> ModelSpec:
    model_id = os.environ.get("ALLM_WRITER_MODEL", "qwen2.5:7b-instruct")
    spec = ModelSpec.from_yaml(ROOT / "configs/models/ollama_exam_writer.yaml")
    return spec.model_copy(update={"model_id": model_id})


def make_exam_generator(samples):
    if os.environ.get("ALLM_EXAM", "dataset").lower() == "generative":
        return ModelExamGenerator(load_model(writer_spec()), difficulty=1, max_attempts=2)
    return DatasetExamGenerator(samples)


def make_grader():
    """Exact match first; escalate chatty wrong-looking answers to LLM judge."""
    judge_model = load_model(grader_spec())
    return EscalatingGrader(
        ExactMatchGrader("contains"),
        LLMJudgeGrader(judge_model, compare_exact=True),
    )


def main() -> None:
    setup_logging("INFO")
    iterations = int(os.environ.get("ALLM_ITERATIONS", "5"))
    workdir = Path(tempfile.mkdtemp(prefix="allm-loop-real-"))
    store = SQLiteRecordStore(workdir / "allm.sqlite3")

    samples = [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, (a, topic)) in enumerate(FACTS.items())
    ]
    grader = make_grader()
    state = KnowledgeState(store)
    teacher = Teacher(
        state,
        make_exam_generator(samples),
        grader,
        TeacherConfig(confidence_smoothing=0.5),
    )

    graph = KnowledgeGraph(store)
    graph.add(Concept(name="symbol-manipulation", description="working with symbols"))
    graph.add(Concept(name="math", prerequisites=("symbol-manipulation",), usefulness=0.9))
    graph.add(Concept(name="geography", prerequisites=("symbol-manipulation",), usefulness=0.6))

    pool = SamplePool()
    pool.ingest(samples)
    memory = EpisodicMemory(store)
    run = LocalTracker(workdir / "runs").start_run("continuous-learning-real")

    student_model = load_model(student_spec())
    students = [
        ModelStudent(sid, "general", student_model)
        for sid in ("alpha", "beta")
    ]

    loop = LearningLoop(
        teacher=teacher,
        students=students,
        planner=NeedPlanner(),
        trainer=InContextTrainer(),
        pool=pool,
        memory=memory,
        failure_log=FailureLog(store),
        graph=graph,
        compression=CompressionEngine(graph),
        debate=DebateEngine(grader=ExactMatchGrader("contains")),
        run=run,
        config=LoopConfig(iterations=iterations, questions_per_exam=6, seed=11),
    )

    print(f"\n=== Real-model loop ({iterations} iterations) ===")
    print(f"  student: {student_spec().model_id}")
    print(f"  grader:  {grader_spec().model_id} ({grader_spec().device})")
    if os.environ.get("ALLM_EXAM", "dataset").lower() == "generative":
        print(f"  exams:   generative ({writer_spec().model_id})")
    else:
        print("  exams:   dataset-backed")

    first_scores: dict[str, float] = {}
    last_scores: dict[str, float] = {}
    for report in loop.run():
        print(f"\niteration {report.iteration}:")
        for s in report.students:
            print(
                f"  {s.student_id}: {s.score_before:.2f} -> {s.score_after:.2f}"
                f"  goals={list(s.goals)}  studied={s.samples_studied}"
            )
            first_scores.setdefault(s.student_id, s.score_before)
            last_scores[s.student_id] = s.score_after

    kel = KnowledgeEvaluationLayer(graph, store, state)
    kel_report = kel.evaluate()
    print("\n=== KEL ===")
    print(f"  learning gain (LG): {kel_report.lg:+.4f}" if kel_report.lg is not None else "  LG: n/a")
    print(f"  concept reuse (CRR): {kel_report.crr}" if kel_report.crr is not None else "  CRR: n/a")
    print(f"  graph stability (GST): {kel_report.gst}" if kel_report.gst is not None else "  GST: n/a")

    print("\n=== Plan.md metrics ===")
    for student in students:
        ev = evaluate_student(state, memory, student.student_id)
        print(
            f"  {ev.student_id}: learning_speed={ev.learning_speed:+.2f} "
            f"mastery={ev.mastery:.2f} self_correction={ev.self_correction_rate}"
        )
        for topic, delta in sorted(ev.improvement_per_topic.items()):
            print(f"    {topic}: {delta:+.2f}")

    comparison = calibration_comparison(state, "alpha")
    print("\n=== Confidence calibration (alpha) ===")
    print(
        f"  primary: n={comparison.primary.n_samples} "
        f"brier={comparison.primary.brier_score:.3f}"
    )
    if comparison.self_reported:
        print(
            f"  self-reported: n={comparison.self_reported.n_samples} "
            f"brier={comparison.self_reported.brier_score:.3f}"
        )
    if comparison.logprob:
        print(
            f"  log-prob: n={comparison.logprob.n_samples} "
            f"brier={comparison.logprob.brier_score:.3f}"
        )

    m1_lg = kel_report.lg is not None and kel_report.lg > 0
    m1_cal = comparison.primary.n_samples > 0
    m1_loop = all(
        last_scores.get(sid, 0) >= first_scores.get(sid, 0)
        for sid in first_scores
    )
    print("\n=== M1 exit criteria ===")
    print(f"  [{'PASS' if m1_loop else 'FAIL'}] loop runs with real model")
    print(f"  [{'PASS' if m1_lg else 'FAIL'}] KEL LG > 0 over {iterations} iterations")
    print(f"  [{'PASS' if m1_cal else 'FAIL'}] calibration report exists")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
