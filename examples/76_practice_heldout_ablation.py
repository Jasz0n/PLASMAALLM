"""M48 exit criterion: practice beats description on HELD-OUT variables.

The generalization form of the founding ablation (PRACTICE.md section
5). Two students, same model, same exam:

- **description-trained** studies only what a book could say about each
  procedure (no outcomes — the rate rules live in the code, not the
  description);
- **practice-trained** studies the outcomes the engine observed for
  *some* variable values.

Both then predict outcomes for variable values **neither has ever
seen**. A student that only memorised cannot answer; a student that
inferred the underlying rule from observed runs can. Ground truth comes
from executing the held-out runs — the students never see it.

    PYTHONPATH=src python3 examples/76_practice_heldout_ablation.py
    ALLM_STUDENT_SIZE=small  ...   # qwen2.5:0.5b instead of 7b
    ALLM_PRACTICE_STUDENT=echo ... # offline machinery check (no Ollama)

Reported through KEL: learning gain per arm plus EGR (every training
run entered the evidence ledger).
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

from allm.core.logging import setup_logging
from allm.evidence import EvidenceLedger
from allm.exam import ExactMatchGrader
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.models import EchoModel, ModelSpec
from allm.models.base import model_loaders
from allm.practice import (
    PracticeProcedure,
    SandboxExecutor,
    VariableSpec,
    description_samples,
    practice_samples,
    prediction_exam,
    record_sweep,
    run_to_package,
)
from allm.practice.types import SweepResult
from allm.storage import SQLiteRecordStore
from allm.students import ModelStudent
from allm.teacher import KnowledgeState, Teacher, TeacherConfig
from allm.trainer import InContextTrainer

# Outcomes are deliberately NOT derivable from the descriptions: the
# rate constants and the offset live only in the programs. Practice is
# the sole source of that knowledge — held-out prediction then tests
# whether the student inferred the rule from observed runs.
CARRIER = PracticeProcedure(
    id="carrier_pricing",
    description=(
        "Compute the shipping price in euros for a parcel of a given weight "
        "in kg, using the carrier's internal rate rules."
    ),
    program="print(int(4 + 3 * weight))\n",
    variables=(VariableSpec(name="weight", default=1, candidates=(2, 4, 8, 10)),),
    topic="logistics",
)

CIPHER = PracticeProcedure(
    id="machine_cipher",
    description=(
        "Encode a lowercase word with the coding machine's internal "
        "letter substitution."
    ),
    program=(
        "print(''.join(chr((ord(c) - 97 + 2) % 26 + 97) for c in word))\n"
    ),
    variables=(
        VariableSpec(name="word", default="cab", candidates=("bad", "ace", "fog", "hi")),
    ),
    topic="codes",
)

HELDOUT = {
    "carrier_pricing": ({"weight": 3}, {"weight": 6}, {"weight": 20}),
    "machine_cipher": ({"word": "dad"}, {"word": "bee"}, {"word": "go"}),
}


def pick_model():
    provider = os.environ.get("ALLM_PRACTICE_STUDENT", "auto")
    if provider != "echo":
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
            size = os.environ.get("ALLM_STUDENT_SIZE", "")
            model_id = os.environ.get(
                "ALLM_STUDENT_MODEL",
                "qwen2.5:0.5b-instruct" if size == "small" else "qwen2.5:7b-instruct",
            )
            spec = ModelSpec(name="practice-ablation", provider="ollama", model_id=model_id)
            return model_loaders.get("ollama")().load(spec), model_id
        except Exception:
            if provider == "auto":
                print("(Ollama unreachable — falling back to echo machinery check)")
            else:
                raise
    spec = ModelSpec(name="practice-ablation", provider="echo", model_id="none")
    return EchoModel(spec), "echo"


def main() -> None:
    setup_logging("WARNING")
    workdir = Path(tempfile.mkdtemp(prefix="allm-practice-heldout-"))
    store = SQLiteRecordStore(workdir / "ablation.sqlite3")
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    state = KnowledgeState(store)
    kel = KnowledgeEvaluationLayer(graph, store, state, ledger=ledger)
    kel.evaluate()  # baseline: EGR measures what THIS experiment earns
    executor = SandboxExecutor()
    grader = ExactMatchGrader()
    procedures = (CARRIER, CIPHER)

    print("=== 1. Practice: observe training runs, earn the evidence ===")
    train_runs = {}
    for procedure in procedures:
        spec = procedure.variables[0]
        runs = tuple(
            executor.run(procedure, {spec.name: value})
            for value in (spec.default, *spec.candidates)
        )
        sweep = SweepResult(
            procedure_id=procedure.id,
            variable=spec.name,
            fixed={},
            runs=runs,
            depends=len({r.outcome for r in runs}) > 1,
        )
        record_sweep(graph, procedure, sweep)
        for run in runs:
            ledger.submit(run_to_package(procedure, run))
        train_runs[procedure.id] = runs
        observed = ", ".join(f"{r.variables[spec.name]!r}->{r.outcome}" for r in runs)
        print(f"{procedure.id}: observed {observed}")

    print("\n=== 2. Held-out ground truth (students never see this) ===")
    heldout_runs = {}
    for procedure in procedures:
        runs = tuple(executor.run(procedure, v) for v in HELDOUT[procedure.id])
        heldout_runs[procedure.id] = runs
        hidden = ", ".join(
            f"{next(iter(v.values()))!r}->{r.outcome}"
            for v, r in zip(HELDOUT[procedure.id], runs)
        )
        print(f"{procedure.id}: {hidden}")

    model, model_id = pick_model()
    print(f"\n=== 3. Two students, same model ({model_id}), same exams ===")
    trainer = InContextTrainer()
    arms = {}
    for arm in ("description", "practice"):
        student = ModelStudent(arm, "procedure outcomes", model)
        samples = (
            description_samples(procedures)
            if arm == "description"
            else [
                s
                for procedure in procedures
                for s in practice_samples(procedure, train_runs[procedure.id])
            ]
        )
        trainer.train(student, samples)
        arms[arm] = student

    teacher = Teacher(state, None, grader, TeacherConfig(confidence_smoothing=1.0))
    results = {}
    for arm, student in arms.items():
        seen_scores, held_scores = [], []
        for procedure in procedures:
            seen = prediction_exam(
                procedure, train_runs[procedure.id], exam_id=f"seen-{arm}-{procedure.id}"
            )
            held = prediction_exam(
                procedure, heldout_runs[procedure.id], exam_id=f"held-{arm}-{procedure.id}"
            )
            seen_scores.append(teacher.evaluate(student, seen).score)
            held_scores.append(teacher.evaluate(student, held).score)
        results[arm] = (
            sum(seen_scores) / len(seen_scores),
            sum(held_scores) / len(held_scores),
        )
        print(
            f"{arm + '-trained':<22} seen {results[arm][0]:.2f}   "
            f"HELD-OUT {results[arm][1]:.2f}"
        )

    print("\n=== 4. KEL verdict ===")
    report = kel.evaluate()
    gap = results["practice"][1] - results["description"][1]
    print(f"learning gain (LG): {report.lg}   evidence growth (EGR): {report.egr}")
    print(f"held-out practice advantage: {gap:+.2f}")
    if model_id == "echo":
        print("echo student: memorisation only, held-out 0.00 expected for both arms —")
        print("run with Ollama for the real generalization verdict.")
    elif gap > 0:
        print("M48 exit criterion MET: the practice-trained student generalized to")
        print("variable values it never observed; the description arm could not.")
    else:
        print("criterion NOT met on this run/model — the gap must be positive.")

    store.close()
    print(f"\nworkdir: {workdir}")


if __name__ == "__main__":
    main()
