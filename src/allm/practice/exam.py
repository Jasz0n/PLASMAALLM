"""Prediction exams: grade against what actually happened (PRACTICE.md).

The student predicts a run's outcome; the expected answer is execution
ground truth. Failed predictions flow into the existing ``FailureLog``
unchanged — "I expected O1, got O2" is the practice engine's training
signal.
"""

from __future__ import annotations

from allm.exam.base import Exam, Question
from allm.data.base import Sample
from allm.practice.types import PracticeProcedure, PracticeRun

PRACTICE_KIND = "practice"


def prediction_prompt(procedure: PracticeProcedure, run: PracticeRun) -> str:
    variables = ", ".join(f"{k}={v!r}" for k, v in sorted(run.variables.items()))
    return (
        f"Predict the outcome of procedure {procedure.id!r} "
        f"({procedure.description}) when {variables}."
    )


def prediction_question(
    procedure: PracticeProcedure, run: PracticeRun, *, question_id: str
) -> Question:
    """One observed run → one gradable prediction question."""
    return Question(
        id=question_id,
        prompt=prediction_prompt(procedure, run),
        expected=run.outcome,
        topic=procedure.concept_name,
        kind=PRACTICE_KIND,
    )


def prediction_exam(
    procedure: PracticeProcedure, runs: tuple[PracticeRun, ...], *, exam_id: str
) -> Exam:
    """An exam over observed runs; expected answers are ground truth."""
    if not runs:
        raise ValueError("cannot build a prediction exam without runs")
    return Exam(
        id=exam_id,
        title=f"outcome prediction: {procedure.id}",
        questions=tuple(
            prediction_question(procedure, run, question_id=f"{exam_id}-q{i}")
            for i, run in enumerate(runs, start=1)
        ),
    )


def practice_samples(
    procedure: PracticeProcedure, runs: tuple[PracticeRun, ...]
) -> list[Sample]:
    """Observed outcomes as study material (PRACTICE.md Stage 6)."""
    return [
        Sample(
            id=f"practice-{run.id}",
            input=prediction_prompt(procedure, run),
            target=run.outcome,
            metadata={"topic": procedure.concept_name, "sample_kind": "practice"},
        )
        for run in runs
    ]


def description_samples(procedures: tuple[PracticeProcedure, ...]) -> list[Sample]:
    """The text-only control arm: what a book could say about a procedure.

    Deliberately outcome-free — the ablation in PRACTICE.md section 5
    measures exactly the knowledge these samples cannot carry.
    """
    return [
        Sample(
            id=f"description-{procedure.id}",
            input=f"What is procedure {procedure.id!r}?",
            target=procedure.description,
            metadata={"topic": procedure.concept_name, "sample_kind": "description"},
        )
        for procedure in procedures
    ]
