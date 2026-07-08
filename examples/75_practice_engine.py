"""PRACTICE.md demo (M48): the second way of learning, end to end.

The same procedure with different variables returns different outcomes
— knowledge no text corpus carries. The engine runs variable sweeps,
turns every execution into an evidence package, writes dependency
relations into the knowledge graph, surfaces a reproduction failure as
an experiment proposal, and then runs the founding ablation: a student
trained on textual descriptions vs one trained on observed outcomes,
sitting the same outcome-prediction exam.

    python examples/75_practice_engine.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.evidence import EvidenceBinder, EvidenceLedger
from allm.exam import ExactMatchGrader
from allm.knowledge import KnowledgeGraph
from allm.models import EchoModel, ModelSpec
from allm.practice import (
    PracticeProcedure,
    SandboxExecutor,
    VariableSpec,
    description_samples,
    next_variable,
    practice_samples,
    prediction_exam,
    record_sweep,
    reproduction_conflict,
    run_sweep,
    run_to_package,
)
from allm.proposals import ProposalBoard
from allm.storage import SQLiteRecordStore
from allm.students import ModelStudent
from allm.trainer import InContextTrainer

COLLATZ = PracticeProcedure(
    id="collatz_steps",
    description="Count how many Collatz iterations a starting integer needs to reach 1.",
    program=(
        "steps = 0\n"
        "n = start\n"
        "while n != 1:\n"
        "    n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
        "    steps += 1\n"
        "print(steps)\n"
    ),
    variables=(
        VariableSpec(name="start", default=6, candidates=(7, 27)),
        VariableSpec(name="verbose", default=0, candidates=(1,)),
    ),
    topic="number-theory",
)

GROWTH = PracticeProcedure(
    id="compound_growth",
    description="Compound an amount over periods at a fixed rate; print the result rounded to 2 places.",
    program="print(round(amount * (1 + rate) ** periods, 2))\n",
    variables=(
        VariableSpec(name="amount", default=100, candidates=(200,)),
        VariableSpec(name="rate", default=0.1, candidates=(0.5,)),
        VariableSpec(name="periods", default=3, candidates=(10,)),
    ),
    topic="finance",
)

FLAKY = PracticeProcedure(
    id="unseeded_random",
    description="Draw one float from an unseeded random generator.",
    program="import random\nprint(random.random())\n",
    variables=(VariableSpec(name="draws", default=1),),
    topic="stochastics",
)


def main() -> None:
    setup_logging("WARNING")
    workdir = Path(tempfile.mkdtemp(prefix="allm-practice-"))
    store = SQLiteRecordStore(workdir / "practice.sqlite3")
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    board = ProposalBoard(store, EvidenceBinder(graph, ledger))
    executor = SandboxExecutor()

    print("=== 1. Curiosity: which variable should we vary first? ===")
    choice = next_variable(COLLATZ, [])
    print(f"least-explored variable of {COLLATZ.id}: {choice!r}")

    print("\n=== 2. Sweeps: same procedure, different variables ===")
    all_runs = []
    for procedure, variable in (
        (COLLATZ, "start"),
        (COLLATZ, "verbose"),
        (GROWTH, "rate"),
        (GROWTH, "periods"),
    ):
        sweep = run_sweep(procedure, variable, executor=executor)
        record_sweep(graph, procedure, sweep)
        all_runs.extend((procedure, run) for run in sweep.runs)
        outcomes = ", ".join(
            f"{run.variables[variable]!r}->{run.outcome}" for run in sweep.runs
        )
        print(f"{procedure.id}: {sweep.relation:<38} ({outcomes})")

    print("\n=== 3. Every run is an evidence package ===")
    for procedure, run in all_runs:
        ledger.submit(run_to_package(procedure, run))
    package = run_to_package(*all_runs[0])
    print(f"{len(all_runs)} packages submitted; example claim:")
    print(f"  {package.claim}")
    print(f"  contributor={package.contributor} outcome={package.outcome}")

    print("\n=== 4. Traceability: concept -> relation -> runs ===")
    concept = graph.get(COLLATZ.concept_name)
    print(f"{concept.name}: related={list(concept.related)}")
    print(f"evidence: {len(concept.evidence)} run(s), e.g. {concept.evidence[0].source} "
          f"({concept.evidence[0].detail})")

    print("\n=== 5. Reproduction failure -> preserved conflict -> proposal ===")
    first = executor.run(FLAKY, {})
    second = executor.run(FLAKY, {})
    conflict = reproduction_conflict(FLAKY, first, second)
    if conflict is None:
        print("the unseeded generator reproduced by chance — rerun the example")
    else:
        proposal = board.from_conflict(conflict)
        print(f"outcomes differed: {first.outcome[:10]}... vs {second.outcome[:10]}...")
        print(f"proposal {proposal.id} [{proposal.status}]: {proposal.question}")

    print("\n=== 6. The founding ablation: description vs practice ===")
    training_runs = tuple(run for _, run in all_runs if run.procedure_id == COLLATZ.id)
    exam = prediction_exam(COLLATZ, training_runs, exam_id="practice-exam-01")
    grader = ExactMatchGrader()
    trainer = InContextTrainer()
    scores = {}
    for arm, samples in (
        ("description-trained", description_samples((COLLATZ,))),
        ("practice-trained", practice_samples(COLLATZ, training_runs)),
    ):
        student = ModelStudent(
            arm, "practice", EchoModel(ModelSpec(name=arm, provider="echo", model_id="none"))
        )
        trainer.train(student, samples)
        graded = [grader.grade(q, student.solve(q)) for q in exam.questions]
        scores[arm] = sum(1 for g in graded if g.correct) / len(graded)
        print(f"{arm:<22} score {scores[arm]:.2f} on {len(graded)} prediction question(s)")

    gap = scores["practice-trained"] - scores["description-trained"]
    print(f"\npractice advantage: +{gap:.2f} — the outcome knowledge was not in the text.")
    print("(With a real model — see Roadmap M48 — the held-out form of this exam")
    print(" tests generalization to variable values the student never observed.)")

    store.close()
    print(f"\nworkdir: {workdir}")


if __name__ == "__main__":
    main()
