# PRACTICE.md — The Practice Engine (ALLM Module)

## Purpose

The Practice Engine gives ALLM the **second way of learning**.

Knowledge is learned two ways: through **text and images** (KDP's
territory) and through **practice** — doing the same thing with
different variables and observing that the outcome changes. Some
knowledge only exists in that difference. A learner built purely on
descriptions of the world is slightly corrupt in exactly the places
where only doing reveals the truth.

The Practice Engine is not a model.

It is a **deterministic act–observe–record system**: it executes
procedures, captures what actually happened, and converts the outcomes
into the same first-class knowledge every other module speaks —
evidence packages, graph relations, exam questions, training samples.

---

# 1. Position in ALLM

```text id="practice-flow-1"
Procedures + Variables → Practice Engine → Runs (ground truth)
        → Evidence Packages → Knowledge Graph → Planner/Teacher/Exam
        → Prediction Exams → FailureLog → Trainer
```

KDP is the only entry point for **unstructured text**. The Practice
Engine is the only entry point for **observed execution outcomes**.

---

# 2. Input Types

* **PracticeProcedure** — a parameterized program (v0: pure Python,
  stdlib only) with named **variables**, each with a declared set of
  candidate values. The procedure's output is its observable outcome.
* Curated procedure catalogs (YAML/code), including procedures derived
  from knowledge units of type `procedure` (KDP → practice).

---

# 3. Output Types

## 3.1 PracticeRun — the ground-truth record

```json id="practice-run-1"
{
  "id": "run_<content-hash>",
  "procedure_id": "collatz_steps",
  "variables": {"start": 27},
  "status": "ok | crash | timeout",
  "outcome": "111",
  "stdout": "111\n",
  "stderr": "",
  "duration_seconds": 0.03,
  "executed_at": "..."
}
```

The run id is content-addressed from (procedure, variables, outcome) —
same experiment, same record.

## 3.2 Evidence package per run

Every run auto-emits an `EvidencePackage` (kind `experiment`,
contributor `practice-engine`): claim *"procedure P with variables V
yields O"*, measurements = variables + outcome, reproduction steps =
the exact program. Practice conclusions are therefore inspectable with
the same machinery as human contributions — nothing hidden.

## 3.3 Graph relations

A **variable sweep** (same procedure, one variable varied, the rest
fixed) yields a dependency verdict:

* outcomes differ → relation `outcome depends on <variable>`
* outcomes identical → relation `outcome independent of <variable>`

Both verdicts are knowledge; both land on the procedure's concept with
the runs as evidence.

## 3.4 Contradictions

Repeating a run with **identical variables** must reproduce the
outcome. When it does not, that is a KDP-style `ConflictNode`
(nondeterminism observed, preserved, never deleted) and becomes an
**experiment proposal** through the existing board — resolved only by
evidence, never by decree.

---

# 4. Pipeline Overview

## Stage 1 — Definition

Procedures are declared with variables, defaults, candidate values and
a topic. Programs are pure Python run with `-I` (isolated) and a hard
timeout; variables are injected as literal bindings, never string
interpolation into code.

## Stage 2 — Curiosity: choose the next experiment

The engine proposes *which variable to vary next*: the least-explored
variable first (fewest distinct observed values), mirroring Plan.md's
curiosity principle — the most informative experiment is where we have
looked least.

## Stage 3 — Execution (sandboxed)

Subprocess isolation plus **kernel resource limits** (M50):
`ResourceLimits` applies POSIX rlimits — CPU seconds, address space,
file size, core dumps off — in the child before `exec`, so the kernel
stops CPU spins, memory bombs and disk-filling that a wall-clock
timeout alone cannot. Repo trials get a roomier budget
(`REPO_TASK_LIMITS`); the `CodingGrader` shares the same guard. Full
OS isolation (container/jail) remains the M50 exit bar for running
anything untrusted. Everything observable is captured; crashes and
timeouts are outcomes, not errors.

## Stage 4 — Recording

Run → evidence package → knowledge graph (append-only, versioned,
provenance preserved). Sweeps write dependency relations; repeats
write contradictions when reproduction fails.

## Stage 5 — Prediction exams

The student must **predict the outcome** of a run: *"What does P
output when start=27?"*. Grading is by execution ground truth, not
string opinion. Failed predictions enter the `FailureLog` exactly like
failed exam answers — "I expected O₁, got O₂" is the practice
engine's training signal.

## Stage 6 — Training samples

Observed outcomes convert to study samples (question → observed
outcome, topic = the procedure's concept), so both trainers
(in-context and LoRA) consume practice knowledge unchanged.

---

# 5. The core experiment (M48 exit criterion)

Two students, same procedures:

* **description-trained** — studies only textual descriptions of the
  procedures (what a book could say);
* **practice-trained** — studies the outcomes the engine observed.

On outcome-prediction exams, the practice-trained student must win.
The gap *is* the thesis, measured: the information was not in the
text. With a real model, the held-out form (predict outcomes for
**unseen variable values**) tests genuine generalization.

---

# 6. Key Design Principles

## 6.1 Ground truth by execution

An outcome is what actually happened, never what a grader believes.

## 6.2 Runs are evidence

Every conclusion traces: concept → relation → evidence package →
captured run. Same traceability rules as human evidence.

## 6.3 Failure is data

Crash and timeout are recorded outcomes with full capture.

## 6.4 Determinism honored, nondeterminism surfaced

Same procedure + variables must reproduce; observed nondeterminism is
a preserved conflict and a proposal, not noise to average away.

## 6.5 Append-only, like everything else

No run, package or relation is ever overwritten.

---

# 6b. Repo-grounded practice (M49)

The engine's procedures extend to real repositories: **the repo's own
test suite is the grader** (`repo_test_procedure` — outcome `pass` /
`fail: <line>`), and a candidate fix is a `CandidatePatch` (full new
content of one repo-relative file) tried in a **disposable copy** of
the repo (`trial_patch`) — the working tree is never touched by a
trial.

A patch that survives its trial becomes a **contribution proposal**
(`ContributionBoard`: `proposed → approved/rejected → applied`), and
the module *is* the invariant:

> **Nothing leaves the system without a human approval record.**

Applying requires an approval carrying a named human reviewer and a
reason; rejections also carry reasons (they become failure samples).
There is deliberately no push, no remote, no auto-merge anywhere —
applying writes one local file under a recorded human approval, and
whatever happens next is a human's git history under a human's name.

**The verdict is the lesson** (`record_review_outcome`): every review
becomes a one-question graded exam in teacher state — approval scores
1.0 and certifies the patch as a studyable expected answer; rejection
scores 0.0 and lands in the `FailureLog` with the reviewer's reason as
feedback (the right answer stays honestly unknown). KEL's learning
gain sees review history exactly like any other measured knowledge.

---

# 7. Non-Goals (v0)

* Hardware or physical experiments (the platform's humans do those;
  their results arrive as evidence packages)
* Network access or shell access inside procedures
* Grading untrusted third-party submissions (blocked on M50 sandboxing)
* Choosing the curriculum (the planner owns priorities; practice
  reports, the Teacher schedules)

---

# 8. Integration with ALLM

| Module | Integration |
|---|---|
| Evidence (`allm.evidence`) | every run auto-emits a package |
| Knowledge graph (Phase 5) | dependency relations + evidence, append-only |
| Proposals (`allm.proposals`) | reproduction failures → `from_conflict` |
| Exam engine (Phase 7) | prediction questions, kind `practice` |
| Students / FailureLog (Phase 3) | failed predictions → training samples |
| Trainer (Phase 3/M2) | practice samples feed both trainers unchanged |
| KEL | learning gain on practice topics = the module's own report card |

---

# 9. Success Criteria

The Practice Engine is correct if:

* every practice conclusion in the graph traces to the runs behind it;
* a variable sweep produces dependency relations a text corpus could
  not contain;
* at least one reproduction failure is detected, preserved as a
  conflict and surfaced as an experiment proposal;
* the practice-trained student beats the description-trained student
  on outcome prediction (Section 5), measured as a KEL-style ablation.

---

# 10. Final Principle

> Text tells the system what people said about the world.
> Practice tells the system what the world did.
