# Roadmap.md — From Architecture to Autonomous Learner

## Where we are (v0.1 — done)

All ten phases of `Plan.md` plus KDP, KEL, evidence packages, proposals
and the HTTP API are implemented: 221 offline tests, nine runnable
examples, every architectural invariant enforced in code (append-only
versioned storage, protocol-based replaceability, evidence-derived
confidence, determinism where specified).

**The honest gap:** everything is validated against deterministic mocks
(echo models, scripted students, synthetic transcripts). The
architecture is proven; the *learning* is not. No real language model
has sat an exam. No weight has been updated. KDP has never digested a
real corpus. That gap defines the roadmap.

**How progress is judged:** by KEL, not by feelings. Every milestone
below has exit criteria expressed in the system's own metrics
(learning gain, concept reuse, graph health, conflict resolution).
We built the measuring instrument first so the roadmap could use it.

---

## M1 — Make it real (v0.2): real models in the loop

*The single biggest de-risking step. Everything else assumes this works.*

| Deliverable | Notes |
|---|---|
| Real-model students end-to-end | `ModelStudent` over small open models through the full loop |
| Log-prob confidence estimator | replaces self-reported confidence where the backend exposes token probabilities; calibration comparison vs self-report |
| LLM-judge grader | `graders` entry for open-ended/reasoning answers (an LLM grades against a rubric); judge disagreement with `ExactMatchGrader` is itself a signal |
| Real generative exams | `ModelExamGenerator` validated against real models; prompt/parse robustness pass |
| Device/dtype hardening | MPS (Apple Silicon), CUDA and CPU paths exercised; graceful degradation |

**Exit criteria**
- The continuous loop (example 06) runs with a real model on one machine.
- KEL learning gain (LG) > 0 over ≥ 5 iterations with in-context learning only.
- Confidence calibration report exists (does stated confidence predict correctness?).

**Risks:** small models may be too weak to show measurable learning →
mitigate with narrow domains and easy exams first; the loop's design
(fresh test exams) already guards against memorisation illusions.

---

## M2 — Real learning (v0.3): weights actually change

*Status: ablation passing on fiction domain — LoRA beats in-context on
weight-only held-out re-exams (notes cleared).*

| Deliverable | Notes |
|---|---|
| LoRA/PEFT trainer | second `trainers` backend behind the existing `Trainer` protocol; trains from failure-derived samples |
| Adapter versioning | student checkpoints stored/versioned like every other belief (which adapter answered which exam is provenance) |
| Forgetting watchdog | after each fine-tune, re-examine previously mastered topics; regression shows up as negative LG per topic in KEL |
| In-context vs weights ablation | same curriculum, both trainers, KEL comparison — the platform's first real experiment on itself |

**Exit criteria**
- Weight-level fine-tuning beats in-context learning on *held-out* exam questions for at least one domain.
- No silent catastrophic forgetting: mastery regressions are detected and reported by KEL.

**Validated by:** `examples/12_trainer_ablation.py` — weight-only held-out re-exam with notes cleared.

---

## M3 — Knowledge at scale (v0.4): the 600-transcript test

*In progress: 22 Kids workshops cleaned; corpus → graph → learning loop wired.*

| Deliverable | Notes |
|---|---|
| KDP embedding stage | pinned-model embedding clustering behind Stage 5 (determinism preserved: fixed model + weights); catches paraphrases with zero lexical overlap |
| Character-precise spans | span mapping through the cleaning stage |
| Vector memory backend | FAISS/Chroma behind `memory_backends`; semantic recall for episodic memory |
| Real-corpus benchmark | a genuinely large corpus (hundreds of transcripts/docs) through KDP; measure RCR, conflict quality, wall-clock |
| PostgreSQL backend | second `storage_backends` entry for multi-process/platform deployments; schema is already trivially portable |

**Done so far:** ASR fix (**19/22** MK, **784** samples, **112** definitions); held-out split + paraphrase exams (`examples/18`); 7B loops; HF LoRA held-out (`examples/17`); **Stage 5 embedding clustering** (`ALLM_KDP_EMBEDDINGS=1`); **noisy concept filter** at extraction (graph clean ratio **37% → 74%**); `sentence-transformers` optional extra.

**Exit criteria**
- Hundreds of documents collapse into thousands (not tens of thousands) of atomic units — healthy RCR without false compression (KEL failure mode stays silent).
- Planner produces a sensible roadmap from the resulting graph with **no access to raw text**.
- Concept naming quality reviewed by a human on a sample (the known Stage 4 weakness either acceptable or fixed by the embedding stage).

---

## M4 — Self-direction (v0.5): the system steers itself

*In progress: KEL-steered held-out loop (`examples/19`) with **LearningStrategy** phases and iteration history JSONL.*

| Deliverable | Notes |
|---|---|
| KEL-steered loop | the loop calls `evaluate()` each iteration; trends and failure modes change behaviour (prioritise unstable high-value areas, raise exam difficulty on mastery, halt on static illusion) |
| Difficulty progression | exam difficulty follows measured mastery per topic |
| Debate with argument exchange | persuasion rounds on top of the existing (uncontaminated) disagreement signal |
| Compression with real probes | `PerformanceProbe` wired to actual exam scores in the loop; abstraction retraction observed in practice |
| Autonomous collection v1 | curated source lists (papers, docs) feeding `SamplePool` with provenance-aware quality scoring — no open web crawling yet |

**Exit criteria**
- The system runs unattended for days: GHS trend upward, all four KEL failure modes monitored, at least one auto-detected and auto-corrected course change in the log.
- Curriculum decisions are explainable end-to-end (roadmap reasons → KEL numbers → evidence).

---

## M5 — Hardening (v0.6): ready for strangers

*Prerequisite for any public deployment; independent of M1–M4 and can start earlier if help arrives.*

| Deliverable | Notes |
|---|---|
| Sandboxed CodingGrader | OS-level isolation (container/subprocess jail) before grading untrusted model or human submissions |
| API hardening | auth hook points (the platform owns identity — the core verifies signatures/tokens it's handed), rate limits, input size caps, pagination |
| Operational surface | structured audit log (already versioned — expose it), backup/restore, storage migrations |
| Release engineering | PyPI package, versioned API contract (OpenAPI published), changelog discipline |
| Docs site | rendered architecture + specs + API reference; CONTRIBUTING already in place |

**Exit criteria**
- An external contributor can deploy the core, submit an evidence package over HTTP, and trace a confidence value to its packages — without talking to us.
- Security review of the evidence-submission and code-grading paths.

---

## M6 — Platform integration (v1.0): SocialFi, later by design

Deliberately last: the core must *work as expected* first, and the
boundary is already built for this moment.

- Freeze the evidence-package wire format as a published spec (platform teams build against it).
- Event/webhook stream (new proposals, resolutions, confidence changes) for the platform frontend.
- Identity/incentive mechanics live entirely in the platform; the core keeps seeing opaque contributor ids — replication-aware confidence is already the anti-gaming primitive (popularity literally cannot move belief).
- Pilot community: one narrow research domain, real contributors, measured by CRE and proposal throughput.

**Exit criterion:** one real contested claim goes through the full public
loop — discussion → KDP → conflict → proposal → independent replications
→ confidence shift — with every step inspectable.

---

## Standing rules (apply to every milestone)

1. **Offline tests stay green and fast.** Real-model runs are examples/benchmarks, never test dependencies.
2. **Every milestone reports itself through KEL.** If a milestone can't state its exit criteria in system metrics, the milestone is underspecified.
3. **Spec-first for new modules** (`<MODULE>.md` → discussion → `src/allm/<module>/`), decision tables in `docs/architecture.md` for everything else.
4. **The invariants are non-negotiable:** nothing overwritten, everything replaceable, confidence earned not set, no placeholder code.

## Sequencing and parallelism

```
M1 ──► M2 ──► M4 ──► M6
 │             ▲
 └──► M3 ──────┘        M5 runs alongside anything, latest before M6
```

M1 is the critical path. M3 (knowledge at scale) only needs M1's
embedding infrastructure, not M2's training. M4 needs both a learner
(M2) and a knowledge base worth steering through (M3).
