"""Practice Engine (PRACTICE.md / Roadmap M48), offline."""

from pathlib import Path

import pytest

from allm.exam import ExactMatchGrader
from allm.exam.base import Answer
from allm.knowledge import KnowledgeGraph
from allm.practice import (
    PRACTICE_CONTRIBUTOR,
    PracticeProcedure,
    PracticeRun,
    SandboxExecutor,
    VariableSpec,
    bind_variables,
    description_samples,
    next_variable,
    practice_samples,
    prediction_exam,
    record_sweep,
    reproduction_conflict,
    run_sweep,
    run_to_package,
)
from allm.evidence import EvidenceBinder, EvidenceLedger
from allm.proposals import ProposalBoard
from allm.storage import SQLiteRecordStore
from allm.students import FailureLog

COLLATZ = PracticeProcedure(
    id="collatz_steps",
    description="Count Collatz iterations from a starting integer.",
    program=(
        "steps = 0\n"
        "n = start\n"
        "while n != 1:\n"
        "    n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
        "    steps += 1\n"
        "print(steps)\n"
    ),
    variables=(
        VariableSpec(name="start", default=6, candidates=(27, 8)),
        VariableSpec(name="unused_flag", default=0, candidates=(1,)),
    ),
    topic="number-theory",
)


@pytest.fixture()
def executor() -> SandboxExecutor:
    return SandboxExecutor()


def test_run_captures_ground_truth(executor: SandboxExecutor) -> None:
    run = executor.run(COLLATZ, {"start": 6})
    assert run.status == "ok"
    assert run.outcome == "8"  # 6→3→10→5→16→8→4→2→1
    assert run.variables == {"start": 6, "unused_flag": 0}
    # content-addressed: the same experiment names the same record
    again = executor.run(COLLATZ, {"start": 6})
    assert again.id == run.id


def test_same_procedure_different_variables_different_outcome(
    executor: SandboxExecutor,
) -> None:
    a = executor.run(COLLATZ, {"start": 6})
    b = executor.run(COLLATZ, {"start": 27})
    assert a.outcome == "8" and b.outcome == "111"
    assert a.outcome != b.outcome


def test_crash_and_timeout_are_outcomes(executor: SandboxExecutor) -> None:
    crasher = PracticeProcedure(
        id="crasher",
        description="Always divides by zero.",
        program="print(1 / d)\n",
        variables=(VariableSpec(name="d", default=0),),
    )
    run = executor.run(crasher, {})
    assert run.status == "crash"
    assert "ZeroDivisionError" in run.outcome

    sleeper = PracticeProcedure(
        id="sleeper",
        description="Sleeps past its budget.",
        program="import time\ntime.sleep(t)\nprint('done')\n",
        variables=(VariableSpec(name="t", default=30),),
        timeout_seconds=0.5,
    )
    run = executor.run(sleeper, {})
    assert run.status == "timeout" and run.outcome == "timeout"


def test_executed_code_cannot_read_parent_secrets(
    executor: SandboxExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent's tokens must not leak into the sandbox (M50 security)."""
    monkeypatch.setenv("ALLM_API_TOKEN", "super-secret-token-123")
    monkeypatch.setenv("OLLAMA_API_KEY", "sk-leak-me")
    probe = PracticeProcedure(
        id="env_probe",
        description="Reports whether it can see the parent's secrets.",
        program=(
            "import os\n"
            "print(os.environ.get('ALLM_API_TOKEN'),"
            " os.environ.get('OLLAMA_API_KEY'),"
            " bool(os.environ.get('PATH')))\n"
        ),
        variables=(),
        topic="security",
    )
    run = executor.run(probe)
    assert run.status == "ok"
    # secrets gone, but PATH survives so the program can run at all
    assert run.outcome == "None None True"


def test_variables_bind_as_literals_never_code() -> None:
    assert bind_variables({"x": 2, "s": "a'b"}) == "s = \"a'b\"\nx = 2"
    with pytest.raises(ValueError, match="not a plain literal"):
        bind_variables({"x": object()})
    with pytest.raises(ValueError, match="invalid variable name"):
        bind_variables({"x; import os": 1})


def test_unknown_variable_rejected(executor: SandboxExecutor) -> None:
    with pytest.raises(ValueError, match="unknown variable"):
        executor.run(COLLATZ, {"nope": 1})


def test_run_emits_traceable_evidence_package(executor: SandboxExecutor) -> None:
    run = executor.run(COLLATZ, {"start": 27})
    package = run_to_package(COLLATZ, run)
    assert package.contributor == PRACTICE_CONTRIBUTOR
    assert package.concept == "practice:collatz_steps"
    assert "yields '111'" in package.claim
    assert package.measurements["variables"] == {"start": 27, "unused_flag": 0}
    assert COLLATZ.program in package.reproduction_steps
    assert package.outcome == "supported"


def test_sweep_finds_dependency_and_independence(executor: SandboxExecutor) -> None:
    depends = run_sweep(COLLATZ, "start", executor=executor)
    assert depends.depends and depends.relation == "outcome-depends-on:start"
    independent = run_sweep(COLLATZ, "unused_flag", executor=executor)
    assert not independent.depends
    assert independent.relation == "outcome-independent-of:unused_flag"


def test_sweep_verdicts_land_in_graph_with_evidence(
    tmp_path: Path, executor: SandboxExecutor
) -> None:
    store = SQLiteRecordStore(tmp_path / "graph.sqlite3")
    try:
        graph = KnowledgeGraph(store)
        record_sweep(graph, COLLATZ, run_sweep(COLLATZ, "start", executor=executor))
        record_sweep(
            graph, COLLATZ, run_sweep(COLLATZ, "unused_flag", executor=executor)
        )
        concept = graph.get("practice:collatz_steps")
        assert concept is not None
        assert "outcome-depends-on:start" in concept.related
        assert "outcome-independent-of:unused_flag" in concept.related
        # every conclusion traceable to the runs behind it
        assert all(e.source.startswith("run_") for e in concept.evidence)
        assert len(concept.evidence) >= 5
    finally:
        store.close()


def test_curiosity_prefers_least_explored_variable(executor: SandboxExecutor) -> None:
    assert next_variable(COLLATZ, []) == "start"  # declaration order tie-break
    history = [executor.run(COLLATZ, {"start": v}) for v in (6, 8, 27)]
    assert next_variable(COLLATZ, history) == "unused_flag"


def test_reproduction_failure_becomes_conflict_and_proposal(tmp_path: Path) -> None:
    stable = PracticeRun.build(
        procedure_id=COLLATZ.id,
        variables={"start": 6},
        status="ok",
        outcome="8",
        stdout="8\n",
        stderr="",
        duration_seconds=0.1,
    )
    flaky = PracticeRun.build(
        procedure_id=COLLATZ.id,
        variables={"start": 6},
        status="ok",
        outcome="9",
        stdout="9\n",
        stderr="",
        duration_seconds=0.1,
    )
    assert reproduction_conflict(COLLATZ, stable, stable) is None
    conflict = reproduction_conflict(COLLATZ, stable, flaky)
    assert conflict is not None and conflict.sources == (stable.id, flaky.id)
    store = SQLiteRecordStore(tmp_path / "proposals.sqlite3")
    try:
        graph = KnowledgeGraph(store)
        binder = EvidenceBinder(graph, EvidenceLedger(store))
        proposal = ProposalBoard(store, binder).from_conflict(conflict)
        assert proposal.concept == "practice:collatz_steps"
        assert proposal.status == "open"
    finally:
        store.close()


def test_prediction_exam_grades_by_ground_truth(executor: SandboxExecutor) -> None:
    runs = tuple(executor.run(COLLATZ, {"start": v}) for v in (6, 27))
    exam = prediction_exam(COLLATZ, runs, exam_id="practice-0001")
    grader = ExactMatchGrader()
    right = grader.grade(
        exam.questions[0],
        Answer(question_id=exam.questions[0].id, text="8", confidence=0.9),
    )
    wrong = grader.grade(
        exam.questions[1],
        Answer(question_id=exam.questions[1].id, text="12", confidence=0.9),
    )
    assert right.correct and not wrong.correct
    assert all(q.kind == "practice" for q in exam.questions)


def test_failed_predictions_become_training_samples(
    tmp_path: Path, executor: SandboxExecutor
) -> None:
    run = executor.run(COLLATZ, {"start": 27})
    exam = prediction_exam(COLLATZ, (run,), exam_id="practice-0002")
    wrong = ExactMatchGrader().grade(
        exam.questions[0],
        Answer(question_id=exam.questions[0].id, text="100", confidence=0.5),
    )
    store = SQLiteRecordStore(tmp_path / "failures.sqlite3")
    try:
        log = FailureLog(store)
        log.record("bench", wrong)
        (sample,) = log.training_samples("bench")
        assert sample.target == "111"
    finally:
        store.close()


def test_practice_vs_description_samples() -> None:
    executor = SandboxExecutor()
    runs = tuple(executor.run(COLLATZ, {"start": v}) for v in (6, 27))
    practice = practice_samples(COLLATZ, runs)
    description = description_samples((COLLATZ,))
    assert [s.target for s in practice] == ["8", "111"]
    # the control arm is deliberately outcome-free
    assert all("8" not in s.target and "111" not in s.target for s in description)
