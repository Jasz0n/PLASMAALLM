"""Kids plasma: in-context vs LoRA ablation on HF student.

Compares notes-based learning against LoRA weight updates. Held-out criterion:
train on workshops 1–12, then weight-only re-exam on workshops 13+ (notes cleared).

Requires ML extras::

    pip install torch transformers peft accelerate datasets
    PYTHONPATH=src python3 examples/15_kids_corpus_graph.py
    PYTHONPATH=src python3 examples/17_kids_trainer_ablation.py

Environment:
    ALLM_SAMPLES=exam|definitions   curriculum pool (default exam)
    ALLM_HF_STUDENT=small|medium    HF model (default small; medium = 1.5B)
    ALLM_HF_MODEL                     override HuggingFace model id
    ALLM_HOLDOUT_AFTER=13             split point (default 13)
    ALLM_SAMPLE_LIMIT                 cap train pool (0 = all)
    ALLM_ITERATIONS                   loop iterations (default 4)
    ALLM_LORA_EPOCHS                  LoRA epochs (default 8)
    ALLM_SAMPLES_PER_ITER             train samples per iteration (default 64)
    ALLM_LORA_BOOTSTRAP=1             LoRA pass on full train pool before loop
    ALLM_BOOTSTRAP=1                  in-context: study full train before loop
    ALLM_PARAPHRASE_EXAM=1            paraphrase held-out exam prompts
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.evaluation import evaluate_student
from allm.exam import DatasetExamGenerator, ExactMatchGrader, ParaphraseExamGenerator
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.kdp.curriculum import load_curriculum_splits
from allm.kdp.holdout import sample_source, workshop_number
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import Concept, KnowledgeGraph
from allm.loop import LearningLoop, LoopConfig
from allm.memory import EpisodicMemory
from allm.models import ModelSpec, load_model
from allm.models.base import GenerationParams
from allm.planner import NeedPlanner
from allm.students import FailureLog, ModelStudent, ModelStudentConfig
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import AdapterStore, ForgettingWatchdog, InContextTrainer, LoRAConfig, LoRATrainer

ROOT = Path(__file__).resolve().parents[1]


def hf_spec() -> ModelSpec:
    size = os.environ.get("ALLM_HF_STUDENT", "small").lower()
    paths = {
        "small": ROOT / "configs/models/student_small.yaml",
        "medium": ROOT / "configs/models/student_1.5b.yaml",
        "1.5b": ROOT / "configs/models/student_1.5b.yaml",
    }
    path = paths.get(size, ROOT / "configs/models/student_small.yaml")
    spec = ModelSpec.from_yaml(path)
    device = os.environ.get("ALLM_DEVICE", "auto")
    overrides: dict = {
        "device": device,
        "generation": GenerationParams(max_new_tokens=96, temperature=0.0, top_p=1.0),
    }
    if custom := os.environ.get("ALLM_HF_MODEL"):
        overrides["model_id"] = custom
    return spec.model_copy(update=overrides)


def holdout_exam_generator(holdout: list[Sample]):
    if os.environ.get("ALLM_PARAPHRASE_EXAM", "0") == "1":
        return ParaphraseExamGenerator(holdout, paraphrase=True)
    return DatasetExamGenerator(holdout)


def weight_only_score(teacher: Teacher, student: ModelStudent, exam_pool: list[Sample]) -> float:
    backup = student.snapshot_notes()
    student.replace_notes({})
    try:
        gen = holdout_exam_generator(exam_pool)
        exam = gen.generate(num_questions=min(16, len(exam_pool)), seed=99)
        return teacher.evaluate(student, exam).score
    finally:
        student.replace_notes(backup)


def lora_config() -> LoRAConfig:
    return LoRAConfig(
        epochs=int(os.environ.get("ALLM_LORA_EPOCHS", "8")),
        repetitions=int(os.environ.get("ALLM_LORA_REPS", "2")),
        learning_rate=float(os.environ.get("ALLM_LORA_LR", "5e-4")),
    )


def maybe_bootstrap(
    trainer_name: str,
    trainer: InContextTrainer | LoRATrainer,
    student: ModelStudent,
    train: list[Sample],
) -> int:
    if trainer_name == "lora" and os.environ.get("ALLM_LORA_BOOTSTRAP", "1") == "1":
        report = trainer.train(student, train)
        return report.samples_used
    if trainer_name == "in_context" and os.environ.get("ALLM_BOOTSTRAP", "0") == "1":
        report = trainer.train(student, train)
        return report.samples_used
    return 0


def run_student(
    student_id: str,
    trainer_name: str,
    workdir: Path,
    train: list[Sample],
    holdout: list[Sample],
) -> dict:
    store = SQLiteRecordStore(workdir / f"{student_id}.sqlite3")
    state = KnowledgeState(store)
    loop_teacher = Teacher(
        state,
        DatasetExamGenerator(train),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )
    heldout_teacher = Teacher(
        state,
        holdout_exam_generator(holdout),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC, description="Kids plasma science"))

    pool = SamplePool()
    pool.ingest(train)
    adapter_root = workdir / "adapters"
    samples_per_iter = int(os.environ.get("ALLM_SAMPLES_PER_ITER", "64"))
    if trainer_name == "lora":
        trainer = LoRATrainer(AdapterStore(store, adapter_root), lora_config())
        student_config = ModelStudentConfig(notes_in_prompt=0, max_notes=1)
    else:
        trainer = InContextTrainer()
        student_config = ModelStudentConfig(notes_in_prompt=8, max_notes=max(len(train), 64))

    model = load_model(hf_spec())
    student = ModelStudent(student_id, DEFAULT_TOPIC, model, student_config)
    before_held = weight_only_score(heldout_teacher, student, holdout)
    bootstrap = maybe_bootstrap(trainer_name, trainer, student, train)

    loop = LearningLoop(
        teacher=loop_teacher,
        students=[student],
        planner=NeedPlanner(),
        trainer=trainer,
        pool=pool,
        memory=EpisodicMemory(store),
        failure_log=FailureLog(store),
        graph=graph,
        forgetting=ForgettingWatchdog(loop_teacher),
        config=LoopConfig(
            iterations=int(os.environ.get("ALLM_ITERATIONS", "4")),
            questions_per_exam=min(8, len(train)),
            samples_per_iteration=min(samples_per_iter, len(train)),
            study_failures=False,
            seed=42,
        ),
    )
    reports = loop.run()
    after_held = weight_only_score(heldout_teacher, student, holdout)
    kel = KnowledgeEvaluationLayer(graph, store, state).evaluate()
    ev = evaluate_student(state, EpisodicMemory(store), student_id)
    store.close()
    return {
        "student_id": student_id,
        "trainer": trainer_name,
        "bootstrap": bootstrap,
        "final_score": reports[-1].students[0].score_after,
        "held_out_before": before_held,
        "held_out_after": after_held,
        "learning_gain": kel.lg,
        "learning_speed": ev.learning_speed,
    }


def workshop_summary(samples: list[Sample]) -> str:
    numbers = sorted({workshop_number(sample_source(s)) for s in samples if workshop_number(sample_source(s))})
    return f"{len(samples)} samples, workshops {numbers}"


def main() -> None:
    setup_logging("INFO")
    try:
        train, holdout = load_curriculum_splits(ROOT)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}. Run examples/15_kids_corpus_graph.py first.") from exc
    limit = int(os.environ.get("ALLM_SAMPLE_LIMIT", "0"))
    if limit > 0:
        train = train[:limit]
    if len(train) < 4 or len(holdout) < 4:
        raise SystemExit(f"Need train/holdout pools (got {len(train)}/{len(holdout)})")

    workdir = Path(tempfile.mkdtemp(prefix="allm-kids-ablation-"))
    pool_name = os.environ.get("ALLM_SAMPLES", "exam")
    print("\n=== Kids plasma trainer ablation (HF, held-out) ===")
    print(f"  pool: {pool_name}")
    print(f"  model: {hf_spec().model_id} on {hf_spec().device}")
    print(f"  train:   {workshop_summary(train)}")
    print(f"  holdout: {workshop_summary(holdout)}")
    print(f"  paraphrase holdout: {os.environ.get('ALLM_PARAPHRASE_EXAM', '0') == '1'}")
    print("  held-out test: weight-only re-exam (notes cleared)")

    results = [
        run_student("kids-notes", "in_context", workdir, train, holdout),
        run_student("kids-lora", "lora", workdir, train, holdout),
    ]
    for row in results:
        print(
            f"\n  {row['student_id']} ({row['trainer']}): "
            f"bootstrap={row['bootstrap']} "
            f"train_loop_final={row['final_score']:.2f} "
            f"holdout_weight-only {row['held_out_before']:.2f}->{row['held_out_after']:.2f} "
            f"LG={row['learning_gain']}"
        )

    lora = next(r for r in results if r["trainer"] == "lora")
    ctx = next(r for r in results if r["trainer"] == "in_context")
    held_pass = lora["held_out_after"] > ctx["held_out_after"]
    print("\n=== M2 kids held-out criterion ===")
    print(f"  [{'PASS' if held_pass else 'FAIL'}] LoRA beats in-context on unseen workshops")
    print(f"\nArtifacts under {workdir}")


if __name__ == "__main__":
    main()
