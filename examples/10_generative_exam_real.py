"""M1 demo: generative exams with a real Ollama model.

Validates :class:`ModelExamGenerator` end-to-end: a writer model produces
an exam, a student model sits it, and an escalating grader scores answers.

Requires::

    ollama serve
    ollama pull qwen2.5:7b-instruct      # exam writer (ALLM_WRITER_MODEL)
    ollama pull qwen2.5:0.5b-instruct    # student (ALLM_STUDENT_MODEL)

    python examples/10_generative_exam_real.py
"""

from __future__ import annotations

import os
from pathlib import Path

from allm.core.logging import setup_logging
from allm.exam import EscalatingGrader, ExactMatchGrader, LLMJudgeGrader, ModelExamGenerator
from allm.models import ModelSpec, load_model
from allm.students import ModelStudent
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def writer_spec() -> ModelSpec:
    model_id = os.environ.get("ALLM_WRITER_MODEL", "qwen2.5:7b-instruct")
    spec = ModelSpec.from_yaml(ROOT / "configs/models/ollama_exam_writer.yaml")
    return spec.model_copy(update={"model_id": model_id})


def student_spec() -> ModelSpec:
    model_id = os.environ.get("ALLM_STUDENT_MODEL", "qwen2.5:0.5b-instruct")
    spec = ModelSpec.from_yaml(ROOT / "configs/models/ollama_student.yaml")
    return spec.model_copy(update={"model_id": model_id})


def grader_spec() -> ModelSpec:
    use_cloud = os.environ.get("ALLM_GRADER", "local").lower() == "cloud"
    name = "ollama_grader_cloud.yaml" if use_cloud else "ollama_grader_local.yaml"
    return ModelSpec.from_yaml(ROOT / "configs/models" / name)


def make_grader():
    judge = load_model(grader_spec())
    return EscalatingGrader(
        ExactMatchGrader("contains"),
        LLMJudgeGrader(judge, compare_exact=True),
    )


def main() -> None:
    setup_logging("INFO")
    topics = os.environ.get("ALLM_TOPICS", "math,geography").split(",")
    num_questions = int(os.environ.get("ALLM_QUESTIONS", "4"))

    store = SQLiteRecordStore(":memory:")
    writer = load_model(writer_spec())
    generator = ModelExamGenerator(writer, difficulty=1, max_attempts=2)
    grader = make_grader()
    teacher = Teacher(
        KnowledgeState(store),
        generator,
        grader,
        TeacherConfig(confidence_smoothing=1.0),
    )

    print("\n=== Generative exam (real models) ===")
    print(f"  writer:  {writer_spec().model_id}")
    print(f"  student: {student_spec().model_id}")
    print(f"  topics:  {topics}")

    exam = teacher.create_exam(topics=topics, num_questions=num_questions)
    print(f"\nGenerated {exam.id} — {len(exam.questions)} questions:")
    for q in exam.questions:
        print(f"  [{q.topic}] {q.prompt}  →  {q.expected!r}")

    student = ModelStudent("gen-student", "general", load_model(student_spec()))
    result = teacher.evaluate(student, exam)
    print(f"\nStudent score: {result.score:.2f} ({sum(r.correct for r in result.results)}/{len(result.results)} correct)")
    for qr in result.results:
        mark = "✓" if qr.correct else "✗"
        fb = f" — {qr.feedback}" if qr.feedback else ""
        print(f"  {mark} {qr.question.prompt[:60]}{fb}")

    store.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
