# Roadmap.md — From Autonomous Learner to Working Colleague

*The previous roadmap (M1–M6, v0.1 era) is archived at
[`docs/Roadmap-v0.md`](docs/Roadmap-v0.md). Its first four milestones are
done; its last two (hardening, platform integration) carry forward here.*

## Where we are (v0.5 — the learner works)

All ten phases of `Plan.md`, KDP, KEL, evidence packages, proposals, the
HTTP API and milestones **M1–M46** are implemented: ~530 offline tests,
74 runnable examples, and every architectural invariant enforced in code
(append-only versioned storage, protocol-based replaceability,
evidence-derived confidence, determinism where specified).

The learning is no longer hypothetical:

- Real Ollama students (0.5B–14B) run the full loop end-to-end.
- LoRA fine-tuning beats in-context notes on **weight-only held-out
  re-exams** (0.88 vs 0.12 on the MK corpus with notes cleared).
- 19/22 kids workshops → 784 training samples → measurable learning
  gain, tracked by KEL, steered by KEL (strategy phases, static-illusion
  halt).
- The Researcher discovers workshops, books (PDF + figures) and video,
  syncs multimodal evidence, distills visual briefs, and recommends —
  never teaches. Teacher review UI gates what students see.
- The lifelong-learning stack (M41–M46) adds retrieval strength, decay
  prediction, maintenance optimization, multi-objective KEL,
  multi-dimensional judging and curriculum diagnostics.

**The honest gaps** (they define this roadmap, the way the mock/real gap
defined the last one):

1. **Knowledge enters only through text and images.** KDP digests
   transcripts, books and frames — but nothing in the system *does*
   anything and learns from what happened. The second way knowledge is
   earned — **practice** — is missing: run the same procedure with
   different variables, observe that the outcome changes, and keep the
   outcome as evidence. Today only `CodingGrader` executes anything, and
   only to grade.
2. **Some capabilities are validated against stubs.** Vision captions
   run on real llava; LiveKit, audio analysis and OCR are largely
   fixture-validated. The capability table doesn't yet distinguish
   "works with a real backend" from "works against the stub".
3. **Drift is showing.** `tests/test_loop.py::test_loop_produces_learning`
   fails: the loop's newer strategy phases study fewer samples per
   iteration than the pre-M4 test expects. One known failure, but the
   suite must be unambiguous for the claims to be.
4. **Nobody else can run this yet.** Hardening (sandboxed code
   execution, API auth, packaging, docs site) and SocialFi platform
   integration were specified in the old roadmap and never executed.
5. **The "software" provider is a fixture.** The vision — an AI that
   takes an active part in open-source development — needs the
   Researcher reading real repositories and the students practicing on
   real tasks, with humans reviewing every outward step.

**How progress is judged:** by KEL, not by feelings. Every milestone
states exit criteria in the system's own metrics. Anything that can't
is underspecified.

---

## The thesis of this roadmap: the second way of learning

Knowledge is learned two ways. The first — **text and images** — the
system already does: KDP, books, workshops, vision, OCR. The second is
**practice**: certain things done the same way but with different
variables return different outcomes, and if a system never acts, that
information simply cannot reach it. A learner built only on descriptions
of the world is slightly corrupt in exactly the places where only doing
reveals the truth.

The architecture was accidentally built for this. An **evidence
package** already *is* the record of one practice run: claim, procedure,
variables (versions, conditions), measurements, outcome. What's missing
is the engine that generates those runs deliberately — vary one
variable, execute, capture, compare — and feeds the deltas back as
training signal. That is M48, and everything after it builds on the
same move: coding tasks first (ground truth is free — tests pass or
they don't), then real open-source work, then the platform where humans
run the experiments the AI cannot.

Spelled out, the loop this roadmap closes is:

```
Provider → Researcher → Teacher → Student → Builder → Experiment
   ↑                                                      │
   └────────── Knowledge Graph ◄── Evidence ◄─────────────┘
```

The **Builder** is the Practice Engine (M48) where the system can act
itself — code, simulations — and the platform's humans (M51) where it
cannot: hardware, prototypes, field measurements. Either way the rule
is the same, and since 2026-07-08 it is *measured*: **documents
propose, evidence disposes** (KEL's `unearned_confidence` detector),
and **Evidence Growth Rate (EGR)** — KEL metric 3.7, an `allm
benchmark` column — asks the question exam scores cannot: did the
ecosystem's knowledge get better-founded, or just bigger? A practice
run earns EGR > 0; pure text ingestion honestly reads 0.

---

## M47 — Consolidation (v0.6): make every claim checkable — **done 2026-07-08**

*Cheap, unblocks everything. A masterclass system's first property is
that its claims survive a stranger's scrutiny.*

| Deliverable | Status |
|---|---|
| Green suite | **done (2026-07-08).** Root cause was a kids-corpus `sample_kinds` default leaking into the generic `LoopConfig`, silently emptying pool collection for plain samples — the fix also repaired examples 06/06_real/12 |
| CI | **done (2026-07-08).** Offline tests on every push (3.12/3.13); 24 verified offline examples smoke-run nightly (`.github/workflows/ci.yml`) |
| State-of-the-system benchmark | **done (2026-07-08).** `allm benchmark` (`allm.benchmarks.system_report`): per-corpus exam scores, KEL LG/GHS/RCR, honest held-out gap; echo default, `--student ollama` opt-in |
| Real-vs-stub capability audit | **done (2026-07-08).** Every `ECOSYSTEM_CAPABILITIES.md` row labeled real / auto / fixture with a reproduction command; fixtures called out as roadmap work (software → M49, LiveKit live-stream soak → M51) |
| Docs alignment | **done (2026-07-08).** README "State of the system" section quotes the benchmark; audit linked |

**Exit criteria**
- `pytest` fully green — **met** (574 offline tests). CI badge on the
  README — **met (2026-07-08):** repo live at
  github.com/Jasz0n/PLASMAALLM, CI running on every push.
- `allm benchmark` emits the state-of-the-system report from a clean
  checkout — **met** (fiction seconds; full three-corpora run distills
  all books, minutes).
- Every capability claim in the docs carries its validation status and a
  reproduction command — **met** (`ECOSYSTEM_CAPABILITIES.md`).

---

## M48 — The Practice Engine (v0.7): learning by doing — **done 2026-07-08**

*The new pillar. Knowledge that only practice can carry enters the
system here. Spec: [`PRACTICE.md`](PRACTICE.md), module: `allm.practice`,
demos: `examples/75` (machinery) and `examples/76` (the real-model
held-out ablation that closed the milestone).*

| Deliverable | Status |
|---|---|
| `allm.practice` module | **done (2026-07-08).** Spec-first: `PRACTICE.md` → `PracticeProcedure` (program + variables) / `PracticeRun` (content-addressed ground truth) |
| Execution sandbox | **done (2026-07-08).** `SandboxExecutor`: subprocess `-I` + timeout, variables bound as literals (never interpolated into code); crash/timeout are recorded outcomes. Hard OS isolation lands in M50 |
| Runs become evidence | **done (2026-07-08).** `run_to_package` — claim "procedure P with variables V yields O", exact program as reproduction steps; reproduction failures become `ConflictNode`s → experiment proposals |
| Variable sweeps | **done (2026-07-08).** `next_variable` (least-explored first), `run_sweep`, `record_sweep` — verdicts land as graph relations `outcome-depends-on:<var>` / `outcome-independent-of:<var>` with per-run evidence |
| Practice-derived training samples | **done (2026-07-08).** Failed predictions flow through the existing `FailureLog` → training samples unchanged |
| Practice exams | **done (2026-07-08).** `prediction_exam` with question kind `practice`; expected answers are execution ground truth |

**Exit criteria**
- On a held-out set of practice tasks, a student trained with the
  practice loop beats the same student trained on textual descriptions
  of the same procedures — **met (2026-07-08)**, in both forms:
  - *Recall* (`examples/75`, echo): practice **1.00** vs description
    **0.00** on seen-variable predictions — the outcome knowledge is
    demonstrably absent from the text.
  - *Generalization* (`examples/76`, real Ollama students, variable
    values **neither arm ever observed**, ground truth by execution):
    held-out advantage **+0.17** (qwen2.5:7b) and **+0.33**
    (qwen2.5:14b) over the description arm's flat **0.00**; the gap
    *scales with model capability* (echo 0.00 → 7B +0.17 → 14B +0.33).
    Both arms score through `Teacher.evaluate`; KEL reports LG and
    EGR 14.0 (every training run entered the ledger). Honest limits:
    the gain comes from the inferable linear-rule domain; the cipher
    domain stayed at 0 for both arms at these scales.
- Every practice conclusion in the graph traces to the runs behind it —
  **met (2026-07-08):** concept → relation → `Evidence(source=run_…)` →
  captured execution, asserted in `tests/test_practice.py`.
- At least one contradiction between two runs detected and turned into
  an experiment proposal — **met (2026-07-08):** reproduction failures
  become preserved `ConflictNode`s → `ProposalBoard.from_conflict`
  (demonstrated live on an unseeded RNG in `examples/75`).

**Risks:** sandbox scope creep — start with pure-Python tasks the
existing `CodingGrader` machinery can already contain; simulated
experiments (parameterized models with measurable outputs) before
anything touching hardware.

---

## M49 — Open-source apprentice (v0.8): practice on real work — **done 2026-07-08**

*The Plasma vision made concrete: the AI takes an active part in
open-source development — as an apprentice, never an autonomous
committer. Completed the day the repo went public, with a real merged
contribution: [PR #2](https://github.com/Jasz0n/PLASMAALLM/pull/2).*

| Deliverable | Notes |
|---|---|
| Real software provider | **done (2026-07-08):** `RepositoryProvider` + `discovery.repository` read a real repo (markdown docs, manifests, module docstrings) through KDP with provenance — demonstrated on ALLM's own codebase (`examples/77`: 81 concepts, 80 preserved conflicts). `repo_history.py` adds commits/diffs (`git log --numstat`, no network), issues (offline JSONL export) and **CI runs** (`gh run list --json` export → evidence: green supports, red challenges, cancelled is inconclusive; in-progress runs are not evidence yet). Dogfooded live: PLASMAALLM ingested its own commits + its own green CI runs, EGR 8.0. Forge sync stays the platform's job — the core never touches the network |
| Codebase knowledge packages | **done (2026-07-08):** concepts with evidence — and the roadmap's phrase made literal: `commit_to_package` turns every merged commit into an **evidence package that already happened** (claim = message, measurements = diffstat, contributor = author, outcome = supported), feeding the ledger so **EGR sees a repository's past like fresh experiments** (`examples/79`: EGR 5.0 from three commits + issues) |
| Practice on repo tasks | **done (2026-07-08):** `repo_test_procedure` runs a repo's own tests as a practice procedure; `trial_patch` tries a `CandidatePatch` in a disposable copy — the working tree is never touched by a trial (`allm.practice.repo_tasks`) |
| Contribution proposals | **done (2026-07-08):** `ContributionBoard` (`proposed → approved/rejected → applied`) with the invariant enforced in code: `apply()` raises `ApprovalError` without a named-human approval record; no push/remote code exists (structurally tested). Live run (`examples/78`): qwen2.5:7b read a failing test, wrote the fix, trial passed, apply was blocked until a maintainer approved — then the repo's tests confirmed it |
| Outcome feedback | **done (2026-07-08):** `record_review_outcome` folds every verdict into teacher state as a one-question graded exam — approval = 1.0 (patch becomes a studyable expected answer), rejection = 0.0 with the reviewer's reason attached as `FailureLog` feedback; KEL's LG sees review history like any exam |

**Exit criteria**
- One real repository ingested end-to-end; planner produces a sensible
  study roadmap for it with no access to raw text (the M3 bar, new
  domain) — **met (2026-07-08):** `inject_package_concepts`
  (`researcher/graph_injection.py`) writes package concepts into the
  graph (confidence capped at 0.6 — documents propose, evidence
  disposes; usefulness scales with KDP stability), and `examples/77`
  ends with the NeedPlanner ranking an 81-concept study curriculum for
  ALLM's own repo from the graph alone.
- One contribution goes proposal → human review → merged, with every
  step inspectable, and the review outcome measurably updates the
  graph — **met (2026-07-08), for real:** issue #1 (a genuine bug: the
  benchmark lost its finished report to a missing output directory) →
  failing regression test by the maintainer → fix authored by
  **qwen2.5:7b** via the apprentice pipeline → trial pass in a
  disposable copy (`contrib_b28f49d6` / `run_a39c93d6`) → PR #2 with
  the full evidence trail → **merged by Jasz0n** → approval recorded on
  the ContributionBoard and folded into teacher state
  (`record_review_outcome`: score 1.00). Every step inspectable on
  GitHub and in the versioned store.
- Invariant holds and is tested — **met (2026-07-08):**
  `tests/test_repo_practice.py` asserts `apply()` without approval
  raises, approval requires a named human + reason, settled reviews
  cannot be silently re-reviewed, and the contribution module contains
  no push/remote/network code at all (checked structurally).

---

## M50 — Hardening (v0.9): ready for strangers

*Carried from the old roadmap's M5, now with more urgency: M48/M49 mean
the system executes code routinely.*

| Deliverable | Notes |
|---|---|
| Sandboxed execution | **done (2026-07-09).** Three layers: (1) **bubblewrap namespace isolation** — read-only root, private /tmp, no network (kernel-enforced, tested against a live local daemon), only the declared workdir writable; auto-detected with loud degradation (`practice/isolation.py`); (2) **kernel rlimits** — CPU / memory / file size / no core dumps in every `SandboxExecutor` and `CodingGrader` child (`practice/limits.py`); (3) timeout + `-I` + literal binding + **`clean_env()` — no inherited secrets** (found leaking during the security review; both paths now pass a minimal env allowlist, regression-tested). Repo trials run fully isolated. Remaining caveat, documented not fixable in-process: same-user uid inside the namespace |
| API hardening | **done (2026-07-09).** `TokenVerifier` hook point (platform owns identity; core verifies what it's handed) with two honest defaults — `AllowAllVerifier` (dev, loudly logged) and `StaticTokenVerifier` (`ALLM_API_TOKEN`, constant-time compare); per-principal token-bucket rate limiting (`ALLM_API_RATE_LIMIT`, 429s); input size caps on every write schema (422/413); pagination on list endpoints. Reads open, every write guarded (`api/security.py`, `tests/test_api_security.py`) |
| Operational surface | **audit + backup/restore done (2026-07-09):** the append-only store *is* the audit log, now readable — `store.audit()`, `allm audit --db`, `GET /audit` (metadata only, paginated); consistent online backups via SQLite's backup API (`allm db backup/verify/restore` — restore verifies first, never clobbers silently, displaced file kept as `.replaced`). **Open:** PostgreSQL backend (needs a server to test against honestly — the schema is one table, the port is a copy job) |
| Release engineering | **done (2026-07-09).** v0.9.0: package builds clean (sdist + wheel, no tests/corpora leaked — verified), full metadata + repo URLs; **`docs/openapi.json` is the published wire contract** (18 paths) with a CI drift guard (`test_published_openapi_contract_is_current`); release runbook in `docs/releasing.md`. Uploading to PyPI is a human act with the maintainer's token |
| System dashboard | **done (2026-07-09).** One read-only view of the whole engine (`GET /dashboard` + `/dashboard/state`; `allm dashboard --db … -o snapshot.html` for a standalone offline copy). KEL scorecard with per-metric sparklines over the recorded time series, live failure-mode findings (the correctness signal), knowledge/evidence/proposal/contribution state, and a per-namespace **population census** built on the new `store.namespaces()` — a wired-but-empty subsystem is the honest "something is missing" signal, read from the data not a checklist. `system_state()` is a pure function of the store (tested offline); the snapshot bakes state into the page so it can be archived and compared over time (`api/dashboard.py`, `tests/test_dashboard.py`) |
| Docs site | rendered architecture + module specs + API reference (the dashboard covers live observability; static reference docs still open) |

**Exit criteria**
- An external contributor deploys the core, submits an evidence package
  over HTTP, and traces a confidence value to its packages — without
  talking to us.
- Security review of the three dangerous paths: evidence submission,
  code grading, practice execution. **Done (2026-07-09):
  `docs/security-review.md`** — assessed all three; the review itself
  found and fixed a live environment-secret leak (parent tokens reachable
  from executed code in every mode) and states the operator setup and the
  one unremovable caveat (same-uid execution).

---

## M51 — Platform integration (v1.0): Plasma SocialFi goes live

*Carried from the old roadmap's M6. Deliberately after hardening. The
boundary was built for this moment; LiveKit discovery (M16–M17) already
speaks the platform's streaming layer.*

| Deliverable | Notes |
|---|---|
| Frozen wire format | evidence-package spec published; platform teams build against it. `docs/openapi.json` (21 paths) already freezes every request/response schema with a CI drift guard; the remaining step is a standalone, versioned evidence-package contract decoupled from the HTTP transport |
| Event stream | **slice 1 done (2026-07-09).** `allm.events.EventLog` — ordered, append-only domain events over the same versioned store, polled by cursor (`GET /events?since=<seq>`, never miss or replay). The API emits `evidence.submitted` + `confidence.changed`, `proposal.opened`, `proposal.resolved`; the dashboard mirrors the feed on its read side (`api/dashboard.py`, `tests/test_events.py`). **Open:** outbound webhook dispatch (outward-facing, so it inherits the human-approval / opt-in posture), plus `contribution.*` events. |
| Incentives stay outside | identity/reward mechanics live entirely in the platform; the core sees opaque contributor ids; replication-aware confidence remains the anti-gaming primitive (popularity cannot move belief) |
| Live workshop loop | LiveKit observation → synced evidence → KDP, on real platform streams instead of fixtures |
| Pilot community | one narrow domain, real contributors, measured by conflict-resolution efficiency, proposal throughput and **EGR** (evidence growth rate — already implemented and tracked, KEL.md 3.7) |

**Kickoff sequencing.** (1) *Event stream v0* — done, the push side of
the live feed pairing with the M50 dashboard's read side. (2) *Frozen
wire format* — publish the standalone evidence-package contract so
platform teams can build without reading our source. (3) *Webhook
dispatch* — outbound delivery of the event stream, opt-in and
approval-gated (it is an outward-facing action). (4) *Live workshop
loop* on real LiveKit streams. (5) *Pilot* against the exit criterion.

**Exit criterion** — unchanged, it was right the first time: one real
contested claim goes through the full public loop — discussion → KDP →
conflict → proposal → independent replications → confidence shift —
with every step inspectable.

---

## M52+ — Scale and lifelong operation (v1.x)

Not yet fully specified — deliberately. What we already know belongs here:

- **Corpus scale:** the 600-transcript test from the old M3 — hundreds
  of documents, healthy RCR, no false compression, acceptable wall-clock.
- **Vector memory backend:** semantic recall behind `memory_backends`
  (planned since Phase 6).
- **Long unattended runs:** weeks, not hours — the M42 decay/maintenance
  optimizer earning its keep, all four KEL failure modes monitored, with
  at least one auto-detected and auto-corrected course change per run.
- **Bigger students:** the loop parity and held-out experiments repeated
  above 7B; multi-node when a single machine stops being the bottleneck.
- **Federation and open collection:** other ALLM instances and curated
  open-web sources — only after the provenance and confidence machinery
  has survived a public pilot (M51).

---

## Standing rules (unchanged — they built everything above)

1. **Offline tests stay green and fast.** Real-model and real-repo runs
   are examples/benchmarks, never test dependencies.
2. **Every milestone reports itself through KEL.** No metric, no
   milestone.
3. **Spec-first for new modules** (`PRACTICE.md` before `allm.practice`),
   decision tables in `docs/architecture.md` for everything else.
4. **The invariants are non-negotiable:** nothing overwritten,
   everything replaceable, confidence earned not set, no placeholder
   code — and from M49 on: **nothing leaves the system without a human
   approval record.**

## Sequencing and parallelism

```
M47 ──► M48 ──► M49 ──► M51 ──► M52+
         │               ▲
         └── (practice    │
              proves      │
              thesis)     │
        M50 runs alongside M48/M49, must land before M51
```

M47 is a short, blocking cleanup — nothing ships on a red suite. M48 is
the critical path: it proves the second way of learning and gives M49
its engine. M50 can start any time (and its sandbox work is a direct
prerequisite for taking M48/M49 public). M51 waits for both, exactly as
platform integration waited last time — the core must work as expected
first.
