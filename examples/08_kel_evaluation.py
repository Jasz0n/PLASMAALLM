"""KEL demo: is the system learning, or only reorganizing information?

Distills two conflicting transcripts (KDP), injects them into the
graph, measures a baseline with KEL, then lets a student study the
contested concepts and measures again — the Graph Health Score and the
conflict-resolution metric show the difference between *having*
knowledge and *improving* it.

    python examples/08_kel_evaluation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.data.base import Sample
from allm.exam import DatasetExamGenerator, ExactMatchGrader
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.storage import SQLiteRecordStore
from allm.students import ScriptedStudent
from allm.teacher import KnowledgeState, Teacher, TeacherConfig

WORKSHOP_1 = """Self-attention is a mechanism that relates every token to
every other token. It computes weights between all pairs, all 100% of them.

Dropout is a regularisation technique that removes random units during
training.
"""

WORKSHOP_2 = """The self attention mechanism is a weighted lookup across all
token pairs.

Dropout is the compression of gradients so that training synchronises
faster across machines.
"""

# Exam material keyed to graph-derived concept names.
FACTS = {
    "What does dropout remove during training?": ("random units", "Dropout"),
    "Is dropout regularisation or gradient compression?": ("regularisation", "Dropout"),
    "What does self-attention relate?": ("tokens", "Self-Attention"),
    "What does self-attention compute between pairs?": ("weights", "Self-Attention"),
}


def show(label: str, report) -> None:
    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    print(f"\n  {label}")
    print(f"    RCR={fmt(report.rcr)}  CD={fmt(report.cd)}  GST={fmt(report.gst)}"
          f"  CRR={fmt(report.crr)}  LG={fmt(report.lg)}  CRE={fmt(report.cre)}")
    print(f"    Graph Health Score: {fmt(report.ghs)}")


def main() -> None:
    setup_logging("WARNING")
    workdir = Path(tempfile.mkdtemp(prefix="allm-kel-"))
    store = SQLiteRecordStore(workdir / "allm.sqlite3")
    graph = KnowledgeGraph(store)
    state = KnowledgeState(store)

    print("=== 1. Ingest knowledge (KDP -> graph) ===")
    documents = DocumentStore(store)
    documents.ingest_text("workshop-1.md", WORKSHOP_1, context="transformers")
    documents.ingest_text("workshop-2.md", WORKSHOP_2, context="transformers")
    result = KDPipeline().distill(documents)
    GraphInjector(graph, store).inject(result)
    print(f"  {len(result.units)} units, {len(result.conflicts)} conflict(s) "
          f"({', '.join(c.concept for c in result.conflicts)})")

    kel = KnowledgeEvaluationLayer(graph, store, state)
    show("Baseline (knowledge exists, nobody has learned anything):", kel.evaluate(result))

    print("\n=== 2. The system studies the contested knowledge ===")
    samples = [
        Sample(id=f"s{i}", input=q, target=a, metadata={"topic": topic})
        for i, (q, (a, topic)) in enumerate(FACTS.items())
    ]
    teacher = Teacher(
        state, DatasetExamGenerator(samples), ExactMatchGrader(),
        TeacherConfig(confidence_smoothing=1.0),
    )
    student = ScriptedStudent("student-1", "transformers")
    teacher.evaluate(student, teacher.create_exam(num_questions=4, seed=1))  # fails
    for question, (answer, _) in FACTS.items():
        student.learn(question, answer)
    teacher.evaluate(student, teacher.create_exam(num_questions=4, seed=2))  # passes
    teacher.assign_goals("student-1")
    print("  student examined twice on Dropout and Self-Attention; learned in between")

    show("After learning (same graph, measurably better):", kel.evaluate())

    print("\n=== 3. Trends (KEL.md section 5) ===")
    for metric in ("lg", "crr", "cre", "ghs"):
        trend = kel.trend(metric)
        print(f"  {metric.upper()}: {'—' if trend is None else f'{trend:+.3f}'}")

    print("\n=== 4. Failure-mode diagnosis (KEL.md section 9) ===")
    findings = kel.diagnose()
    if not findings:
        print("  no failure modes detected")
    for finding in findings:
        print(f"  {finding.mode}: {finding.detail}")

    store.close()
    print(f"\nDone. Time series preserved under {workdir}")


if __name__ == "__main__":
    main()
