"""Kids plasma: held-out learning loop (train workshops 1–12, test 13+).

Honest evaluation: the student studies **train** samples only; every exam
draws questions from **held-out** workshops the loop never pools for study.

    PYTHONPATH=src python3 examples/18_kids_heldout_loop.py

Environment (same as example 16, plus):
    ALLM_HOLDOUT_AFTER      first held-out workshop number (default 13)
    ALLM_SAMPLE_KIND        filter kinds (e.g. definition,we_call)
    ALLM_SAMPLES=definitions  use samples_definitions.jsonl
    ALLM_PARAPHRASE_EXAM=1   rephrase holdout prompts (tests understanding)
    ALLM_BOOTSTRAP            0 recommended (default 0 here)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.evaluation import diagnose_holdout_gap, evaluate_student, format_holdout_gap_report
from allm.exam import (
    DatasetExamGenerator,
    EscalatingGrader,
    ExactMatchGrader,
    LLMJudgeGrader,
    ParaphraseExamGenerator,
)
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.kdp.curriculum import load_curriculum_splits
from allm.kdp.holdout import sample_source, workshop_number
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import LearningLoop, LoopConfig
from allm.memory import EpisodicMemory
from allm.models import ModelSpec, load_model
from allm.planner import NeedPlanner
from allm.students import FailureLog, ModelStudent, ModelStudentConfig
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import InContextTrainer

ROOT = Path(__file__).resolve().parents[1]


def student_spec() -> ModelSpec:
    size = os.environ.get("ALLM_STUDENT_SIZE", "large").lower()
    config_name = "ollama_student.yaml" if size == "small" else "ollama_student_7b.yaml"
    default_model = "qwen2.5:0.5b-instruct" if size == "small" else "qwen2.5:7b-instruct"
    model_id = os.environ.get("ALLM_STUDENT_MODEL", default_model)
    spec = ModelSpec.from_yaml(ROOT / "configs/models" / config_name)
    return spec.model_copy(update={"model_id": model_id})


def grader_spec() -> ModelSpec:
    use_cloud = os.environ.get("ALLM_GRADER", "local").lower() == "cloud"
    name = "ollama_grader_cloud.yaml" if use_cloud else "ollama_grader_local.yaml"
    return ModelSpec.from_yaml(ROOT / "configs/models" / name)


def exam_generator(holdout):
    if os.environ.get("ALLM_PARAPHRASE_EXAM", "0") == "1":
        return ParaphraseExamGenerator(holdout, paraphrase=True)
    return DatasetExamGenerator(holdout)


def workshop_summary(samples) -> str:
    numbers = sorted({workshop_number(sample_source(s)) for s in samples if workshop_number(sample_source(s))})
    if not numbers:
        return f"{len(samples)} samples"
    return f"workshops {numbers[0]}–{numbers[-1]} ({len(numbers)} files, {len(samples)} samples)"


def main() -> None:
    setup_logging("INFO")
    try:
        train, holdout = load_curriculum_splits(ROOT)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}. Run examples/15_kids_corpus_graph.py first.") from exc
    if len(train) < 4 or len(holdout) < 4:
        raise SystemExit(f"Need train and holdout pools (got {len(train)}/{len(holdout)})")

    iterations = int(os.environ.get("ALLM_ITERATIONS", "8"))
    questions = int(os.environ.get("ALLM_QUESTIONS", "8"))
    samples_per_iter = int(os.environ.get("ALLM_SAMPLES_PER_ITER", "64"))
    max_notes = int(os.environ.get("ALLM_MAX_NOTES", str(max(len(train), 256))))
    notes_in_prompt = int(os.environ.get("ALLM_NOTES_IN_PROMPT", "16"))

    workdir = Path(tempfile.mkdtemp(prefix="allm-kids-heldout-"))
    store = SQLiteRecordStore(workdir / "kids.sqlite3")
    state = KnowledgeState(store)
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC, description="Kids plasma science"))

    judge = LLMJudgeGrader(load_model(grader_spec()), compare_exact=True)
    grader = EscalatingGrader(ExactMatchGrader("contains"), judge)
    teacher = Teacher(
        state,
        exam_generator(holdout),
        grader,
        TeacherConfig(confidence_smoothing=0.5),
    )

    pool = SamplePool()
    pool.ingest(train)
    student = ModelStudent(
        "kids-heldout",
        DEFAULT_TOPIC,
        load_model(student_spec()),
        ModelStudentConfig(max_notes=max_notes, notes_in_prompt=notes_in_prompt),
    )
    trainer = InContextTrainer()
    bootstrap = 0
    if os.environ.get("ALLM_BOOTSTRAP", "0") == "1":
        bootstrap = trainer.train(student, train).samples_used

    loop = LearningLoop(
        teacher=teacher,
        students=[student],
        planner=NeedPlanner(),
        trainer=trainer,
        pool=pool,
        memory=EpisodicMemory(store),
        failure_log=FailureLog(store),
        graph=graph,
        config=LoopConfig(
            iterations=iterations,
            questions_per_exam=min(questions, len(holdout)),
            samples_per_iteration=min(samples_per_iter, len(train)),
            study_failures=False,
            seed=42,
        ),
    )

    paraphrase = os.environ.get("ALLM_PARAPHRASE_EXAM", "0") == "1"
    kind_filter = os.environ.get("ALLM_SAMPLE_KIND", "") or (
        "definitions" if os.environ.get("ALLM_SAMPLES", "").lower() == "definitions" else "all"
    )
    print("\n=== Kids held-out learning loop ===")
    print(f"  train:   {workshop_summary(train)}")
    print(f"  holdout: {workshop_summary(holdout)}")
    print(f"  filter:  {kind_filter}")
    print(f"  paraphrase exams: {paraphrase}")
    print(f"  student: {student_spec().model_id}")
    print(f"  bootstrap train only: {bootstrap}")
    print(f"  iterations: {iterations}  questions/exam: {loop._config.questions_per_exam}")
    gap = diagnose_holdout_gap(train, holdout)
    print("\n=== Hold-out curriculum gap (pre-loop) ===")
    print(format_holdout_gap_report(gap))

    first_score = last_score = 0.0
    for report in loop.run():
        row = report.students[0]
        first_score = first_score or row.score_before
        last_score = row.score_after
        print(
            f"\n  iter {report.iteration}: "
            f"{row.score_before:.2f} -> {row.score_after:.2f} "
            f"studied={row.samples_studied}"
        )

    kel = KnowledgeEvaluationLayer(graph, store, state).evaluate()
    ev = evaluate_student(state, EpisodicMemory(store), student.student_id)
    print("\n=== Held-out outcomes ===")
    print(f"  holdout exam: {first_score:.2f} -> {last_score:.2f}")
    print(f"  learning_speed: {ev.learning_speed:+.3f}  mastery: {ev.mastery:.2f}")
    if kel.lg is not None:
        print(f"  KEL learning gain: {kel.lg:+.4f}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
