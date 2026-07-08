# ALLM — Autonomous Learning Language Model

[![CI](https://github.com/Jasz0n/PLASMAALLM/actions/workflows/ci.yml/badge.svg)](https://github.com/Jasz0n/PLASMAALLM/actions/workflows/ci.yml)

An experimental research platform exploring autonomous learning in
language models: a system that evaluates its own knowledge, plans its
own curriculum, studies, examines itself and compresses what it learns.

ALLM is also the knowledge engine of an open, evidence-based research
platform ([`smallVision.md`](smallVision.md)): humans contribute
discussions and **evidence packages**, the AI organizes, finds
contradictions and proposes what to test next, humans verify — and the
shared knowledge base measurably improves. Its central principle:

> Knowledge is earned through transparent evidence, not authority.
> Confidence increases through reproducible results, not popularity.
> Every conclusion remains traceable to the observations behind it.

- **Vision & spec:** [`Plan.md`](Plan.md) (source of truth)
- **Roadmap:** [`Roadmap.md`](Roadmap.md) (where this is heading, with exit criteria)
- **Platform vision:** [`smallVision.md`](smallVision.md)
- **Module specs:** [`KDP.md`](KDP.md), [`KEL.md`](KEL.md), [`PRACTICE.md`](PRACTICE.md)
- **Architecture & decisions:** [`docs/architecture.md`](docs/architecture.md)
- **Contributing:** [`CONTRIBUTING.md`](CONTRIBUTING.md) · MIT licensed

## Status

- **Phase 1 (infrastructure) — done.** Configuration, logging, experiment
  tracking, CLI, plugin registry, dependency injection, versioned storage,
  model loading, dataset loading. No training yet, by design.
- **Phase 2 (teacher) — done.** Exam vocabulary + dataset-backed exam
  generation + deterministic grading (`allm.exam`), the `Student`
  protocol with a scripted double (`allm.students`), and the `Teacher`
  API: evaluate, assign goals from weaknesses, measure progress, all
  persisted with full belief history (`allm.teacher`).

- **Phase 3 (students) — done.** `ModelStudent` wraps any
  `LanguageModel`, studies facts into a bounded note memory, answers
  from memory or via the model with self-reported confidence, and every
  failure is stored versioned and convertible back into training
  samples (`FailureLog`). The `Trainer` protocol ships with an
  in-context backend and a LoRA/PEFT backend for Hugging Face models.

- **M1 (v0.2) — real models in the loop — done.** Ollama-backed
  students, LLM-judge grading, log-prob confidence, generative exams,
  and KEL learning-gain validation. See [`Roadmap.md`](Roadmap.md).

- **M2 (v0.3) — real learning — done.** LoRA trainer with
  exam-aligned prompts and answer-only loss masking, versioned adapter
  store, forgetting watchdog, and trainer ablation
  (`examples/12_trainer_ablation.py`). LoRA reaches loop parity with
  in-context and **beats it on weight-only held-out re-exams** (notes
  cleared) — fiction domain on Qwen2.5-0.5B.

- **Phase 4 (planner) — done.** The need-based planner implements
  Plan.md's curiosity engine (`Need = Weakness × Importance × Curiosity
  × Novelty`) with prerequisite-aware ordering: blocked topics wait,
  and their urgency flows to the prerequisites that unblock them.
  Signals merge the teacher's measured state with a YAML topic catalog.

- **Phase 5 (knowledge graph) — done.** Versioned concepts with
  prerequisites, relations, confidence, usefulness, curiosity and
  append-only evidence; cycles rejected at write time; every revision
  needs a reason and full history survives. `to_catalog()` turns the
  graph into the planner's curriculum.

- **Phase 6 (memory) — done.** Append-only episodic memory (successes,
  failures, revisions, reasoning traces) with filtered recall and
  lexical search; `remember_exam` turns graded exams into episodes
  automatically. Vector recall is a planned second backend.

- **Phase 7 (exam engine) — done.** `ModelExamGenerator` lets any
  language model write exams (factual/reasoning/coding/cross-domain,
  difficulty-controlled) in a strictly parsed format; `CodingGrader`
  executes Python submissions in a subprocess with a timeout; and
  `CompositeGrader` routes mixed exams to the right grader per
  question.

- **Phase 8 (debate) — done.** Students answer the same question
  independently; answers cluster by normalised text, disagreement is
  measured, verdicts are confidence-weighted (and graded when ground
  truth exists), and unresolved debates convert into learning tasks.

- **Phase 9 (compression) — done.** Concepts resting on identical
  foundations are abstracted into higher-level principles that carry
  the union of their members' evidence; members survive, linked to the
  principle. A performance probe retracts (never deletes) abstractions
  that hurt predictive performance.

- **Phase 10 (loop) — done.** The deduplicating `SamplePool` collector,
  derived evaluation metrics (improvement, learning speed, mastery,
  self-correction), and the `LearningLoop` composition root running the
  full cycle: measure → plan → collect → learn → debate → test →
  compress → update memory → repeat.

- **KDP (knowledge distillation pipeline) — done.** A deterministic
  staged compiler (see [`KDP.md`](KDP.md)) from raw transcripts/notes to
  atomic, deduplicated, provenance-carrying knowledge units — with
  contradiction detection — injected append-only into the knowledge
  graph. `allm kdp distill notes/*.md --db experiments/allm.sqlite3`.

- **KEL (knowledge evaluation layer) — done.** Seven epistemic metrics
  (see [`KEL.md`](KEL.md)): redundancy collapse, conflict density,
  graph stability, concept reuse, learning gain, conflict-resolution
  efficiency, and **evidence growth rate** (did the knowledge get
  better-founded, not just bigger?) — composed into a Graph Health
  Score, tracked as persisted time series with trends, plus failure-mode
  detectors (false compression, dead knowledge growth, conflict
  accumulation, static illusion, **unearned confidence** — documents
  propose, evidence disposes). Measurement-only: answers "is ALLM
  learning, or only reorganizing information?"

- **Evidence packages — done.** Contributions as structured, traceable
  evidence (claim, artifacts, measurements, reproduction steps,
  outcome) with replication-aware, popularity-resistant confidence:
  fifty posts from one lab count once; one independent replication
  counts a lot. Every confidence value ships with its full breakdown.
- **Experiment proposals — done.** Debate disagreements, KDP conflicts
  and planner gaps become proposals (`open → claimed → resolved`);
  resolution happens only through evidence packages — never by decree.
- **HTTP API — done.** The platform boundary (`pip install -e ".[api]"`):
  submit evidence and documents, browse concepts with provenance,
  drive the proposal lifecycle, take KEL measurements.

- **Researcher (v0) — in progress.** Discover workshops + software fixture,
  build Knowledge Packages, enqueue recommendations for Teacher/planner.
  See [`RESEARCHER.md`](RESEARCHER.md) and [`ResearcherPlan.md`](../ResearcherPlan.md).

- **M48 (v0.7) — Practice Engine core — done.** The second way of
  learning ([`PRACTICE.md`](PRACTICE.md)): sandboxed procedure runs with
  variables bound as literals, every execution an evidence package,
  variable sweeps writing `outcome-depends-on:<var>` relations into the
  graph, reproduction failures becoming experiment proposals, and
  outcome-prediction exams graded by execution ground truth. The
  founding ablation (`examples/75`): practice-trained **1.00** vs
  description-trained **0.00** — the outcome knowledge was not in the
  text. Exit criterion closed by the real-model held-out ablation
  (`examples/76`): on variable values *neither student ever observed*,
  practice-trained scores **+0.17** (qwen2.5:7b) / **+0.33**
  (qwen2.5:14b) over the description arm's flat **0.00** — the
  generalization gap scales with model capability.

- **M49 (v0.8) — open-source apprentice — in progress.** The Researcher
  reads **real repositories** (markdown docs, manifests, module
  docstrings through KDP with provenance; `examples/77` studies ALLM's
  own codebase — 81 concepts, 80 preserved conflicts). Repo-grounded
  practice: the repo's own test suite is the grader; candidate patches
  are trialed in disposable copies and become **contribution proposals**
  gated by the tested invariant *nothing leaves the system without a
  human approval record* (`examples/78`: qwen2.5:7b fixed a real failing
  test end-to-end — trial passed, apply blocked until a named maintainer
  approved), and every review verdict folds back into teacher state as
  a graded exam (approval = studyable answer; rejection = failure
  sample carrying the reviewer's reason). Repo *history* is evidence
  that already happened: commits (claim + diffstat + author + merged
  outcome), exported issues, and CI runs (green supports, red
  challenges) feed the ledger and the graph — EGR measures a
  repository's past like fresh experiments (`examples/79`; dogfooded on
  this repo's own history and CI). Open: the first real external merged
  contribution.

- **M41–M46 (lifelong learning stack) — done.** KS planner + retrieval strength,
  decay prediction + maintenance optimizer, multi-objective KEL, multi-dimensional
  grading (curriculum / alignment / evidence judges), KEL → Researcher remediation
  requests, and Researcher model router + curriculum diagnostics (Chief Scientist).
  Capstone examples: `examples/70_m42_decay_optimizer_kel.py` … `examples/74_m46_curriculum_diagnostics_kel.py`.

**All ten phases plus KDP, KEL, evidence, proposals and the API are
implemented.** M1 adds real Ollama models end-to-end; M2 adds weight-level
LoRA training. M3–M4 (Kids corpus, KEL-steered loop) are done. M41–M46 extend
the stack toward stable, adaptive lifelong learning — see
[`Roadmap.md`](Roadmap.md) and [`docs/architecture.md`](docs/architecture.md).

## State of the system — one command

```bash
PYTHONPATH=src python3 -m allm.cli.main benchmark                 # fiction + kids + books, offline echo student
PYTHONPATH=src python3 -m allm.cli.main benchmark --student ollama  # real model (needs Ollama)
PYTHONPATH=src python3 -m allm.cli.main benchmark --corpora fiction --iterations 2  # quick smoke (~seconds)
```

Emits the M47 report — exam scores, KEL learning gain, EGR (evidence
growth rate), GHS, held-out gap, RCR — per standard corpus
(`allm benchmark --output report.json` for the JSON). Offline reference
numbers (echo student, seed 13): fiction LG **1.00**, kids LG **0.50**,
books LG **0.35** / RCR **0.70**, practice LG **1.00** / **EGR 12.00**
(the only corpus that *earns* its evidence — text corpora honestly read
EGR 0), held-out **0.00** everywhere — an echo student memorises and
must not appear to generalize; real-model numbers come from
`--student ollama`. The full run distills all three books and takes
minutes; `--corpora fiction,practice` takes seconds.

Every capability claim in
[`ECOSYSTEM_CAPABILITIES.md`](ECOSYSTEM_CAPABILITIES.md) now carries a
validation label (**real** / **auto** / **fixture**) with the command
that reproduces it. CI runs the offline test suite on every push and
smoke-runs 24 offline examples nightly (`.github/workflows/ci.yml`).

## Current model roles (Ollama local)

| Role | Model | Notes |
|------|--------|--------|
| **Student** | `qwen2.5:7b-instruct` (default) | `ALLM_STUDENT_SIZE=small` → 0.5B; override `ALLM_STUDENT_MODEL` |
| **Grader / Judge** | `qwen2.5:14b-instruct` | Multi-dimensional exam judge when `ALLM_MULTI_JUDGE=1` |
| **Teacher** | *(no LLM)* | Exam sampling, curriculum, visual approval — orchestration only |
| **Researcher (reasoning)** | `qwen2.5:14b-instruct` | Curriculum diagnostics, conflict triage — `ALLM_RESEARCHER_REASONING_MODEL` |
| **Researcher (vision)** | `llava` | Book figure caption + OCR — `ALLM_VISION_MODEL` / `ALLM_OCR_MODEL` |

Pull vision before multimodal book runs: `ollama pull llava`.

## Books-only KEL capstone (M45 / M46)

Train on Keshe books 1–2, hold out book 3, no workshop transcripts:

```bash
cd PLASMAALLM
# M45 — KEL research requests → Researcher remediation
PYTHONPATH=src ALLM_STUDENT_SIZE=large python3 examples/73_m45_kel_research_requests_kel.py

# M46 — adds model router + curriculum diagnostics (Chief Scientist)
PYTHONPATH=src ALLM_STUDENT_SIZE=large python3 examples/74_m46_curriculum_diagnostics_kel.py
```

Key env flags (set by examples 70–74):

| Flag | Purpose |
|------|---------|
| `ALLM_BOOKS_ONLY=1` | Books-only curriculum; no workshop packages |
| `ALLM_KEL_PHASE_ORDER=books_only` | 8 book iterations, no workshop phase |
| `ALLM_VISION_BACKEND=auto` | Real llava captions when Ollama has the model |
| `ALLM_KEL_RESEARCH_REQUESTS=1` | KEL submits remediation tasks to Researcher |
| `ALLM_CURRICULUM_DIAGNOSTICS=1` | Researcher diagnoses *why* learning fails (M46) |
| `ALLM_RESEARCHER_MODEL_ROUTER=1` | Route tasks to reasoning / verifier / vision specialists |

See [`ECOSYSTEM_CAPABILITIES.md`](ECOSYSTEM_CAPABILITIES.md) for the full capability map.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # core + tests (lightweight)
pip install -e ".[ml]"         # optional: torch/transformers/datasets
```

### Kids workshop transcripts (22 files)

Raw ASR exports live in [`transcripts/Kids/`](transcripts/Kids/) (timestamps,
speaker tags, AV troubleshooting). **Train on cleaned full text — not raw files,
not LLM digests.**

```bash
PYTHONPATH=src python3 examples/13_kids_transcripts_kdp.py
```

| Output | Purpose |
|--------|---------|
| `transcripts/Kids/cleaned/*.txt` | Full workshop dialogue (all speakers) |
| `transcripts/Kids/cleaned/mk/*.txt` | Mr Keshe only — every teaching word kept |

KDP distillation (knowledge units for the graph) runs in the same step. Raw
originals are never modified.

Optional: `ALLM_INJECT_GRAPH=1` injects units into the knowledge graph.

**Digests (optional, lossy):** skip for now — use cleaned text only.

**Next — graph + training samples (M3):**

```bash
PYTHONPATH=src python3 examples/15_kids_corpus_graph.py
ALLM_KDP_EMBEDDINGS=1 PYTHONPATH=src python3 examples/15_kids_corpus_graph.py  # Stage 5 embedding clustering
ALLM_KDP_EMBEDDINGS=1 ALLM_KDP_EMBED_BACKEND=st PYTHONPATH=src python3 examples/15_kids_corpus_graph.py  # sentence-transformers
```

Builds the knowledge graph from `cleaned/mk/` (KDP) and exports:

- `transcripts/Kids/samples.jsonl` — full MK pool (definition + we_call + compact Q→A)
- `transcripts/Kids/samples_exam.jsonl` — same exam-friendly kinds (no legacy long-hook teaching)

**Run the loop on that curriculum:**

```bash
# Local Ollama student — 7B instruct (default for kids plasma loops).
ALLM_SAMPLES=exam ALLM_SAMPLE_LIMIT=12 ALLM_ITERATIONS=2 PYTHONPATH=src python3 examples/16_kids_learning_loop.py

# Full run with bootstrap + 7B student (default)
ALLM_SAMPLES=exam PYTHONPATH=src python3 examples/16_kids_learning_loop.py

# Smaller/faster student
ALLM_STUDENT_SIZE=small PYTHONPATH=src python3 examples/16_kids_learning_loop.py
```

**Held-out Ollama loop** (`examples/18`) — honest train 1–12 / test 13+.

**KEL-steered loop** (`examples/19`) — strategy phases (definitions → relations →
reasoning → research), `learning_history.jsonl`, marginal gain per phase,
mission-aware specialist identity; safer static-illusion halt.

**Specialist students** (`configs/students/`) — mission YAML, ingest routing,
expert lookup (`examples/21`). Identity ablation (`examples/22`).

**LoRA vs in-context on the same curriculum (HF student, weight-only held-out):**

```bash
pip install torch transformers peft accelerate datasets
ALLM_SAMPLES=exam PYTHONPATH=src python3 examples/17_kids_trainer_ablation.py          # all ~513 samples
ALLM_HF_STUDENT=medium ALLM_SAMPLES=definitions PYTHONPATH=src python3 examples/17_kids_trainer_ablation.py  # 1.5B LoRA
ALLM_SAMPLE_LIMIT=32 ALLM_SAMPLES=exam PYTHONPATH=src python3 examples/17_kids_trainer_ablation.py  # quick
```

On 24 MK samples, LoRA held-out **0.88** beats in-context notes **0.12** after notes are cleared.
Use `ALLM_SAMPLES=exam` for short prompts. Corpus: **19/22** workshops → **784** samples (after ASR speaker fix).

```bash
# Default: 7B + bootstrap (pre-study full pool, then loop)
ALLM_SAMPLES=exam PYTHONPATH=src python3 examples/16_kids_learning_loop.py

# Definitions-only held-out (cleaner generalization test)
ALLM_SAMPLES=definitions ALLM_BOOTSTRAP=0 PYTHONPATH=src python3 examples/18_kids_heldout_loop.py

# Same with paraphrased exam questions
ALLM_SAMPLES=definitions ALLM_PARAPHRASE_EXAM=1 ALLM_BOOTSTRAP=0 PYTHONPATH=src python3 examples/18_kids_heldout_loop.py

# KEL-steered held-out loop (M4) — strategy phases + learning_history.jsonl
ALLM_SAMPLES=exam ALLM_ITERATIONS=8 ALLM_KEL_STRATEGY_MASTERY=0.35 ALLM_KEL_STRATEGY_WINDOW=3 PYTHONPATH=src python3 examples/19_kids_kel_steered_loop.py

# Disable specialist mission (ablation control arm)
ALLM_STUDENT_IDENTITY=0 ALLM_LOOP_SEED=42 PYTHONPATH=src python3 examples/19_kids_kel_steered_loop.py

# Analyze learning history / marginal gain per strategy phase
PYTHONPATH=src python3 examples/20_analyze_learning_history.py /path/to/learning_history.jsonl

# Specialist routing demo (offline)
PYTHONPATH=src python3 examples/21_specialist_routing.py

# Identity ablation: mission on vs off (same ALLM_LOOP_SEED)
ALLM_ITERATIONS=4 ALLM_LOOP_SEED=42 PYTHONPATH=src python3 examples/22_identity_ablation.py

# Dual specialist vs generalist on mixed plasma + software corpus
PYTHONPATH=src python3 examples/23_dual_specialist_ablation.py --dry-run
ALLM_ITERATIONS=3 ALLM_PEER_CONSULT=1 PYTHONPATH=src python3 examples/23_dual_specialist_ablation.py

# Peer consultation demo (offline)
PYTHONPATH=src python3 examples/24_peer_consultation.py

# Researcher cycle — discover, package, recommend (offline)
PYTHONPATH=src python3 examples/25_researcher_cycle.py

# Teacher-mediated specialist consultation (offline)
PYTHONPATH=src python3 examples/26_mediated_consultation.py

# Researcher → KEL → Student integrated loop (offline)
PYTHONPATH=src python3 examples/27_researcher_kel_loop.py

# Researcher → KEL ecosystem metrics (offline)
PYTHONPATH=src python3 examples/28_researcher_kel_metrics.py

# Dual specialist loop with Teacher-mediated consultation (offline)
PYTHONPATH=src python3 examples/29_mediated_dual_loop.py

# Research plan + capability pipeline (offline)
PYTHONPATH=src python3 examples/31_research_plan_cycle.py

# Per-student ecosystem targeting (offline)
PYTHONPATH=src python3 examples/32_ecosystem_targeting.py

# Researcher brain: curiosity, gaps, missions, tiers (M6)
PYTHONPATH=src python3 examples/33_researcher_brain_cycle.py

# Multimodal sync: video timeline fixture + transcript (M7)
PYTHONPATH=src python3 examples/34_multimodal_sync.py

# Debate + Teacher show me evidence (M8)
PYTHONPATH=src python3 examples/35_debate_show_me.py

# Multimodal learning loop with debate evidence (M9)
PYTHONPATH=src python3 examples/36_multimodal_learning_loop.py

# Auto-generate video fixtures from transcripts (M10)
PYTHONPATH=src python3 examples/37_auto_video_fixtures.py

# Dual specialist + multimodal debate evidence (M10)
ALLM_RESEARCHER=1 ALLM_MULTIMODAL=1 ALLM_DEBATE_EVIDENCE=1 ALLM_CONSULT_SHOW_ME=1 PYTHONPATH=src python3 examples/38_dual_multimodal_loop.py

# Mediated consultation + Teacher show me (M11)
PYTHONPATH=src python3 examples/39_consult_show_me.py

# Vision caption enrichment on synced evidence (M12)
ALLM_VISION_CAPTIONS=1 PYTHONPATH=src python3 examples/40_vision_caption_enrichment.py

# Ollama vision captions — auto-detect daemon (M13)
ALLM_VISION_BACKEND=auto ALLM_VISION_CAPTIONS=1 PYTHONPATH=src python3 examples/41_ollama_vision_caption.py

# Kids workshop → kids-plasma topic alignment demo (offline)
PYTHONPATH=src python3 examples/30_kids_topic_alignment.py
ALLM_MEDIATED_CONSULT=1 ALLM_ITERATIONS=3 PYTHONPATH=src python3 examples/23_dual_specialist_ablation.py --dry-run

# KEL loop with Researcher-boosted planner (Ollama; set ALLM_RESEARCHER=1)
ALLM_RESEARCHER=1 ALLM_ITERATIONS=4 PYTHONPATH=src python3 examples/19_kids_kel_steered_loop.py

# LoRA vs in-context on held-out workshops (HF) — definitions recommended
ALLM_SAMPLES=definitions ALLM_LORA_BOOTSTRAP=1 PYTHONPATH=src python3 examples/17_kids_trainer_ablation.py
```

## Try it

```bash
pytest                                   # run the test suite
python examples/01_infrastructure_tour.py        # offline end-to-end tour
python examples/02_teacher_evaluates_students.py # teacher/student cycle
python examples/03_students_learn_from_failure.py # closed learning cycle
python examples/04_learning_roadmap.py           # prioritised roadmap
python examples/05_knowledge_graph.py            # evolving concept graph
python examples/06_continuous_learning_loop.py   # the full loop, offline
python examples/06_continuous_learning_loop_real.py  # M1: Ollama models
python examples/10_generative_exam_real.py       # M1: generative exam
python examples/11_generative_learning_loop_real.py  # M1: generative loop
python examples/12_trainer_ablation.py         # M2: in-context vs LoRA (weight-only held-out)
python examples/13_kids_transcripts_kdp.py     # M3: clean 22 Kids workshops
python examples/15_kids_corpus_graph.py        # M3: graph + samples.jsonl from cleaned/mk
python examples/16_kids_learning_loop.py       # M3: learn on samples.jsonl (Ollama)
python examples/17_kids_trainer_ablation.py    # M3: kids in-context vs LoRA (HF, held-out)
python examples/18_kids_heldout_loop.py       # M3: honest held-out Ollama loop
python examples/19_kids_kel_steered_loop.py   # M4: KEL-steered strategy phases
python examples/20_analyze_learning_history.py # marginal gain per strategy phase
python examples/21_specialist_routing.py    # specialist ingest routing (offline)
python examples/22_identity_ablation.py     # mission on vs off ablation
python examples/23_dual_specialist_ablation.py  # dual specialist vs generalist
python examples/24_peer_consultation.py       # peer expert consultation (offline)
python examples/25_researcher_cycle.py        # Researcher discover/package/recommend
python examples/26_mediated_consultation.py   # Teacher-mediated specialist consult
python examples/27_researcher_kel_loop.py     # Researcher → KEL → Student (offline)
python examples/28_researcher_kel_metrics.py  # Researcher ecosystem → KEL diagnose
python examples/30_kids_topic_alignment.py   # workshop concepts → kids-plasma pool
python examples/31_research_plan_cycle.py  # L0–L7 capability pipeline demo
python examples/32_ecosystem_targeting.py  # per-student recommendation targeting
python examples/33_researcher_brain_cycle.py  # observe → gap → missions → plan
python examples/34_multimodal_sync.py  # video fixture + transcript sync (M7)
python examples/35_debate_show_me.py  # debate + show me evidence (M8)
python examples/36_multimodal_learning_loop.py  # loop + debate evidence (M9)
python examples/37_auto_video_fixtures.py  # auto video fixtures (M10)
python examples/38_dual_multimodal_loop.py  # dual specialist + multimodal debate
python examples/39_consult_show_me.py  # mediated consult + show me (M11)
python examples/40_vision_caption_enrichment.py  # vision captions (M12)
python examples/41_ollama_vision_caption.py  # Ollama vision captions (M13)
python examples/70_m42_decay_optimizer_kel.py   # M42 decay + maintenance optimizer
python examples/73_m45_kel_research_requests_kel.py  # M45 KEL → Researcher remediation
python examples/74_m46_curriculum_diagnostics_kel.py   # M46 model router + diagnostics
python examples/75_practice_engine.py           # M48: learning by doing (PRACTICE.md)
python examples/76_practice_heldout_ablation.py # M48: held-out ablation (Ollama; echo fallback)
python examples/77_repository_researcher.py     # M49: Researcher reads a real repo (its own)
python examples/78_apprentice_contribution.py   # M49: fix → trial → human-approved apply
python examples/79_repo_history_evidence.py     # M49: commits + issues as evidence (EGR)
python examples/07_kdp_distillation.py           # transcripts -> graph
python examples/08_kel_evaluation.py             # is it actually learning?
python examples/09_evidence_and_proposals.py     # the human-AI evidence loop

# the platform-facing API
pip install -e ".[api]"
uvicorn --factory allm.api.app:create_default_app
allm benchmark --corpora fiction         # state-of-the-system report (M47)
allm info                                # CLI: version + plugins
allm plugins                             # registered backends
allm config show -c configs/allm.yaml    # resolved configuration
allm model validate configs/models/echo.yaml
```

## Layout

```
configs/          project + model/dataset specs (YAML)
configs/students/ specialist identity YAML (mission, domains, shared core)
docs/             architecture notes
examples/         runnable, offline demonstrations
experiments/      run outputs + storage (gitignored)
scripts/          maintenance / one-off scripts
src/allm/core     config, logging, registry, DI container
src/allm/storage  append-only versioned record store (sqlite)
src/allm/tracking experiment runs (local files)
src/allm/models   LanguageModel protocol; echo, ollama + huggingface loaders
src/allm/data     Sample/DatasetSpec; jsonl + huggingface loaders
src/allm/cli      `allm` command-line interface
src/allm/exam     Question/Answer/Exam types, generation, grading
src/allm/students Student protocol, ModelStudent, FailureLog, StudentIdentity
src/allm/teacher  Teacher API + KnowledgeState + expert lookup
src/allm/trainer  Trainer protocol; in-context + LoRA backends
src/allm/planner  need-based roadmap; mission weights; ingest router
src/allm/knowledge versioned concept graph -> planner catalog
src/allm/memory   episodic memory: recall filters + lexical search
src/allm/debate   disagreement measurement -> learning tasks
src/allm/compression evidence-preserving abstraction
src/allm/collector deduplicating, quality-scoring sample pool
src/allm/evaluation Plan.md metrics; holdout gap; MLG; ablation compare
src/allm/kdp      knowledge distillation: raw docs -> knowledge units
src/allm/practice learning by doing: sandboxed runs -> evidence -> relations
src/allm/kel      epistemic metrics, health score, failure detection
src/allm/evidence evidence packages + replication-aware confidence
src/allm/proposals experiment proposals resolved by evidence
src/allm/api      FastAPI boundary for the platform
src/allm/researcher discover → package → recommend; model router + curriculum
                 diagnostics (M46); never teaches students directly
src/allm/loop     KEL-steered loop, learning history, composition root
tests/            pytest suite (runs without ML extras)
```

## Principles

Everything is replaceable (protocols + registries + DI), nothing is
ever overwritten (versioned storage with reasons), and the whole loop
must run offline (deterministic `echo` model provider) so orchestration
research never waits on GPUs.
