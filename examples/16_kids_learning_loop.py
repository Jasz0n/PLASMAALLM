"""Kids plasma curriculum: learning loop on samples.jsonl.

Loads training/exam samples from ``transcripts/Kids/samples.jsonl``
(produced by ``examples/15_kids_corpus_graph.py``), runs the full
measure → plan → collect → learn → test cycle with a real Ollama student
and LLM-judge grading for open-ended plasma answers.

Prerequisites::

    PYTHONPATH=src python3 examples/13_kids_transcripts_kdp.py
    PYTHONPATH=src python3 examples/15_kids_corpus_graph.py

Run::

    ollama serve
    ollama pull qwen2.5:7b-instruct       # student (default)
    ollama pull qwen2.5:14b-instruct    # local judge
    PYTHONPATH=src python3 examples/16_kids_learning_loop.py

Environment:
    ALLM_STUDENT_MODEL   Ollama student tag (default qwen2.5:7b-instruct)
    ALLM_STUDENT_SIZE    small|large — pick ollama_student.yaml vs ollama_student_7b.yaml
    ALLM_GRADER          local|cloud (default local 14b judge)
    ALLM_ITERATIONS      loop iterations (default 5)
    ALLM_QUESTIONS       questions per exam (default 8)
    ALLM_SAMPLE_LIMIT    cap pool size for quick runs (default 0 = all)
    ALLM_SAMPLES         exam — use samples_exam.jsonl (recommended)
    ALLM_SAMPLES_FILE    jsonl path override
    ALLM_BOOTSTRAP       1 — study full pool before first exam (default 1)
    ALLM_BOOTSTRAP_LIMIT cap bootstrap samples (0 = all)
    ALLM_SAMPLES_PER_ITER  pool samples per iteration (default 32)
    ALLM_MAX_NOTES       note store size (default 512 for bootstrap)
    ALLM_NOTES_IN_PROMPT notes shown in model prompt (default 16)
    ALLM_INJECT_GRAPH=1  distill cleaned/mk into graph before loop
    ALLM_TRAINER         in_context (default) or lora (HF student only)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.evaluation import evaluate_student
from allm.exam import DatasetExamGenerator, EscalatingGrader, ExactMatchGrader, LLMJudgeGrader
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kdp.corpus import DEFAULT_TOPIC, ingest_cleaned_corpus, load_samples_jsonl
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
SAMPLES = ROOT / "transcripts" / "Kids" / "samples.jsonl"
SAMPLES_EXAM = ROOT / "transcripts" / "Kids" / "samples_exam.jsonl"
CLEANED_MK = ROOT / "transcripts" / "Kids" / "cleaned" / "mk"


def samples_path() -> Path:
    override = os.environ.get("ALLM_SAMPLES_FILE", "").strip()
    if override:
        return Path(override)
    if os.environ.get("ALLM_SAMPLES", "").lower() == "exam":
        return SAMPLES_EXAM
    return SAMPLES


def student_spec() -> ModelSpec:
    size = os.environ.get("ALLM_STUDENT_SIZE", "large").lower()
    config_name = "ollama_student.yaml" if size == "small" else "ollama_student_7b.yaml"
    default_model = "qwen2.5:0.5b-instruct" if size == "small" else "qwen2.5:7b-instruct"
    model_id = os.environ.get("ALLM_STUDENT_MODEL", default_model)
    spec = ModelSpec.from_yaml(ROOT / "configs/models" / config_name)
    return spec.model_copy(update={"model_id": model_id})


def student_config(sample_count: int) -> ModelStudentConfig:
    max_notes = int(os.environ.get("ALLM_MAX_NOTES", str(max(sample_count, 512))))
    notes_in_prompt = int(os.environ.get("ALLM_NOTES_IN_PROMPT", "16"))
    return ModelStudentConfig(max_notes=max_notes, notes_in_prompt=notes_in_prompt)


def grader_spec() -> ModelSpec:
    use_cloud = os.environ.get("ALLM_GRADER", "local").lower() == "cloud"
    name = "ollama_grader_cloud.yaml" if use_cloud else "ollama_grader_local.yaml"
    return ModelSpec.from_yaml(ROOT / "configs/models" / name)


def load_curriculum_samples() -> list[Sample]:
    path = samples_path()
    if not path.is_file():
        raise SystemExit(
            f"Missing {path}. Run examples/15_kids_corpus_graph.py first."
        )
    samples = [s for s in load_samples_jsonl(path) if s.target]
    limit = int(os.environ.get("ALLM_SAMPLE_LIMIT", "0"))
    if limit > 0:
        samples = samples[:limit]
    if not samples:
        raise SystemExit("No labelled samples in samples.jsonl")
    return samples


def maybe_inject_graph(store: SQLiteRecordStore, graph: KnowledgeGraph) -> None:
    if os.environ.get("ALLM_INJECT_GRAPH", "0") != "1":
        return
    if not CLEANED_MK.is_dir():
        return
    documents = DocumentStore(store)
    ingest_cleaned_corpus(documents, CLEANED_MK)
    result = KDPipeline().distill(documents)
    GraphInjector(graph, store).inject(result)
    print(f"  KDP injected {len(result.units)} units into graph")


def make_trainer(store: SQLiteRecordStore, student: ModelStudent):
    if os.environ.get("ALLM_TRAINER", "in_context").lower() == "lora":
        from allm.trainer import AdapterStore, LoRAConfig, LoRATrainer

        adapter_root = Path(tempfile.mkdtemp(prefix="allm-kids-lora-")) / "adapters"
        return LoRATrainer(AdapterStore(store, adapter_root), LoRAConfig(epochs=4))
    return InContextTrainer()


def maybe_bootstrap_study(
    trainer: InContextTrainer,
    student: ModelStudent,
    samples: list[Sample],
) -> int:
    """Pre-load the curriculum into study memory before the first exam."""
    if os.environ.get("ALLM_BOOTSTRAP", "1") != "1":
        return 0
    limit = int(os.environ.get("ALLM_BOOTSTRAP_LIMIT", "0"))
    batch = samples[:limit] if limit > 0 else samples
    report = trainer.train(student, batch)
    return report.samples_used


def main() -> None:
    setup_logging("INFO")
    iterations = int(os.environ.get("ALLM_ITERATIONS", "5"))
    questions = int(os.environ.get("ALLM_QUESTIONS", "8"))
    samples_per_iter = int(os.environ.get("ALLM_SAMPLES_PER_ITER", "32"))
    samples = load_curriculum_samples()
    sample_path = samples_path()

    workdir = Path(tempfile.mkdtemp(prefix="allm-kids-loop-"))
    store = SQLiteRecordStore(workdir / "kids.sqlite3")
    state = KnowledgeState(store)
    graph = KnowledgeGraph(store)
    graph.add(Concept(name=DEFAULT_TOPIC, description="Kids Knowledge Seekers plasma science"))
    maybe_inject_graph(store, graph)

    judge = LLMJudgeGrader(load_model(grader_spec()), compare_exact=True)
    grader = EscalatingGrader(ExactMatchGrader("contains"), judge)
    teacher = Teacher(
        state,
        DatasetExamGenerator(samples),
        grader,
        TeacherConfig(confidence_smoothing=0.5),
    )

    pool = SamplePool()
    pool.ingest(samples)
    student_model = load_model(student_spec())
    student = ModelStudent(
        "kids-student",
        DEFAULT_TOPIC,
        student_model,
        student_config(len(samples)),
    )
    trainer = make_trainer(store, student)
    bootstrap_count = 0
    if isinstance(trainer, InContextTrainer):
        bootstrap_count = maybe_bootstrap_study(trainer, student, samples)

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
            questions_per_exam=min(questions, len(samples)),
            samples_per_iteration=min(samples_per_iter, len(samples)),
            seed=42,
        ),
    )

    print("\n=== Kids plasma learning loop ===")
    print(f"  samples: {len(samples)} from {sample_path.name}")
    print(f"  student: {student_spec().model_id} (notes max={student._config.max_notes})")
    print(f"  grader:  {grader_spec().model_id} (LLM judge on paraphrases)")
    print(f"  bootstrap: {bootstrap_count} samples pre-studied")
    print(f"  iterations: {iterations}  questions/exam: {loop._config.questions_per_exam}")

    first_score = last_score = 0.0
    for report in loop.run():
        row = report.students[0]
        first_score = first_score or row.score_before
        last_score = row.score_after
        print(
            f"\n  iter {report.iteration}: "
            f"{row.score_before:.2f} -> {row.score_after:.2f} "
            f"studied={row.samples_studied} goals={list(row.goals)}"
        )

    kel = KnowledgeEvaluationLayer(graph, store, state).evaluate()
    ev = evaluate_student(state, EpisodicMemory(store), student.student_id)
    print("\n=== Outcomes ===")
    print(f"  exam score: {first_score:.2f} -> {last_score:.2f}")
    print(f"  learning_speed: {ev.learning_speed:+.3f}  mastery: {ev.mastery:.2f}")
    if kel.lg is not None:
        print(f"  KEL learning gain: {kel.lg:+.4f}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
