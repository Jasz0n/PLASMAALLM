"""M2 demo: in-context vs LoRA trainer ablation on a Hugging Face student.

Compares two students on the same curriculum — one learns via notes
(:class:`InContextTrainer`), one via LoRA weights (:class:`LoRATrainer`).

M2 held-out criterion: after training, evaluate on curriculum questions
with in-context notes **cleared** so only weight-level knowledge counts.
In-context learning should collapse; LoRA should retain facts in weights.

Requires ML extras::

    pip install torch transformers peft accelerate datasets
    PYTHONPATH=src python3 examples/12_trainer_ablation.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.evaluation import evaluate_student
from allm.exam import DatasetExamGenerator, ExactMatchGrader
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
FACTS = {
    "What is the code name of Project Zeta?": ("Nebula", "fiction"),
    "How many moons orbit Planet Xerion?": ("7", "fiction"),
    "What element powers the Stellar Gate?": ("Quantium", "fiction"),
    "Who founded the Lunar Archive?": ("Dr. Venn", "fiction"),
}


def hf_spec() -> ModelSpec:
    path = ROOT / "configs/models/student_small.yaml"
    spec = ModelSpec.from_yaml(path)
    device = os.environ.get("ALLM_DEVICE", "auto")
    return spec.model_copy(
        update={
            "device": device,
            "generation": GenerationParams(max_new_tokens=32, temperature=0.0, top_p=1.0),
        }
    )


def samples_from(facts: dict[str, tuple[str, str]]) -> list[Sample]:
    return [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, (a, topic)) in enumerate(facts.items())
    ]


def weight_only_score(teacher: Teacher, student: ModelStudent, facts: dict) -> float:
    """Held-out: same curriculum questions but in-context notes disabled."""
    backup = student.snapshot_notes()
    student.replace_notes({})
    try:
        exam = DatasetExamGenerator(samples_from(facts)).generate(num_questions=len(facts))
        return teacher.evaluate(student, exam).score
    finally:
        student.replace_notes(backup)


def lora_config() -> LoRAConfig:
    return LoRAConfig(
        epochs=int(os.environ.get("ALLM_LORA_EPOCHS", "8")),
        repetitions=int(os.environ.get("ALLM_LORA_REPS", "2")),
        learning_rate=float(os.environ.get("ALLM_LORA_LR", "5e-4")),
    )


def run_student(student_id: str, trainer_name: str, workdir: Path) -> dict:
    store = SQLiteRecordStore(workdir / f"{student_id}.sqlite3")
    curriculum = samples_from(FACTS)
    state = KnowledgeState(store)
    teacher = Teacher(
        state,
        DatasetExamGenerator(curriculum),
        ExactMatchGrader("contains"),
        TeacherConfig(confidence_smoothing=0.5),
    )
    graph = KnowledgeGraph(store)
    graph.add(Concept(name="basics"))
    graph.add(Concept(name="fiction", prerequisites=("basics",), usefulness=0.9))

    pool = SamplePool()
    pool.ingest(curriculum)
    adapter_root = workdir / "adapters"
    if trainer_name == "lora":
        trainer = LoRATrainer(AdapterStore(store, adapter_root), lora_config())
        student_config = ModelStudentConfig(notes_in_prompt=0, max_notes=1)
    else:
        trainer = InContextTrainer()
        student_config = ModelStudentConfig(notes_in_prompt=4, max_notes=32)

    model = load_model(hf_spec())
    student = ModelStudent(student_id, "fiction", model, student_config)
    before_held = weight_only_score(teacher, student, FACTS)

    loop = LearningLoop(
        teacher=teacher,
        students=[student],
        planner=NeedPlanner(),
        trainer=trainer,
        pool=pool,
        memory=EpisodicMemory(store),
        failure_log=FailureLog(store),
        graph=graph,
        forgetting=ForgettingWatchdog(teacher),
        config=LoopConfig(
            iterations=int(os.environ.get("ALLM_ITERATIONS", "2")),
            questions_per_exam=4,
            seed=42,
        ),
    )
    reports = loop.run()
    after_held = weight_only_score(teacher, student, FACTS)
    kel = KnowledgeEvaluationLayer(graph, store, state).evaluate()
    ev = evaluate_student(state, EpisodicMemory(store), student_id)
    adapters = AdapterStore(store, adapter_root).history(student_id)
    adapter_ids = {
        r.adapter_id
        for r in state.exam_results(student_id)
        if r.adapter_id is not None
    }
    store.close()
    return {
        "student_id": student_id,
        "trainer": trainer_name,
        "final_score": reports[-1].students[0].score_after,
        "held_out_before": before_held,
        "held_out_after": after_held,
        "learning_gain": kel.lg,
        "learning_speed": ev.learning_speed,
        "adapters": len(adapters),
        "adapter_ids": sorted(adapter_ids),
        "forgetting": [
            fr.regressions
            for report in reports
            for fr in report.forgetting
            if fr.regressions
        ],
    }


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-ablation-"))
    print("\n=== M2 trainer ablation (HF model) ===")
    print(f"  model: {hf_spec().model_id} on {hf_spec().device}")
    print(f"  curriculum: {len(FACTS)} fiction facts")
    print("  held-out: weight-only re-exam (notes cleared)")

    results = [
        run_student("hf-notes", "in_context", workdir),
        run_student("hf-lora", "lora", workdir),
    ]
    for row in results:
        print(
            f"\n  {row['student_id']} ({row['trainer']}): "
            f"loop_final={row['final_score']:.2f} "
            f"weight-only {row['held_out_before']:.2f}->{row['held_out_after']:.2f} "
            f"LG={row['learning_gain']} adapters={row['adapters']}"
        )
        if row.get("adapter_ids"):
            print(f"    exam adapter provenance: {row['adapter_ids']}")
        if row["forgetting"]:
            print(f"    forgetting probes: {row['forgetting']}")

    lora = next(r for r in results if r["trainer"] == "lora")
    ctx = next(r for r in results if r["trainer"] == "in_context")
    print("\n=== M2 exit criteria ===")
    held_pass = lora["held_out_after"] > ctx["held_out_after"]
    forget_ok = not ctx["forgetting"] or all(
        abs(v) < 0.5 for reg in ctx["forgetting"] for v in reg.values()
    )
    print(f"  [{'PASS' if held_pass else 'FAIL'}] LoRA beats in-context weight-only held-out")
    print(f"  [{'PASS' if forget_ok else 'WARN'}] no severe forgetting on in-context path")
    if held_pass:
        print("\nM2 complete for fiction domain: weights beat notes when notes are unavailable.")
    print(f"\nArtifacts under {workdir}")


if __name__ == "__main__":
    main()
