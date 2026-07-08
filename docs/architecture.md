# ALLM Architecture

`Plan.md` (project root) is the source of truth for *what* we are
building. This document records *how* the codebase is organised and why.

## Guiding constraints

- **Research platform, not production.** Optimise for swapping
  components and inspecting state, not throughput.
- **Everything replaceable.** Components depend on protocols
  (`typing.Protocol`), get collaborators injected, and concrete
  implementations are chosen by name from registries via configuration.
- **No hidden state.** Config objects are frozen; services are wired
  explicitly in a composition root (the DI container); loggers and
  registries are the only module-level objects.
- **Nothing is ever overwritten.** The storage layer is append-only and
  versioned, with a `reason` per write. Plan.md's memory rules are
  enforced at the lowest layer so later phases cannot violate them by
  accident.
- **Runs without GPUs.** Heavy dependencies (torch, transformers,
  datasets) are optional extras imported lazily. The `echo` model
  provider is a deterministic stand-in so orchestration logic (teacher,
  students, debate, exams) is developed and tested offline.

## Layer map

```
allm.core        config | logging | registry | container       (no internal deps)
      ▲
allm.storage     versioned record store (sqlite; postgres later)
allm.tracking    experiment runs (local files; hosted later)
allm.models      LanguageModel/ModelLoader protocols (echo, huggingface)
allm.data        Sample/DatasetLoader protocols (jsonl, huggingface)
      ▲
allm.cli         argparse front-end over the layers above
      ▲
allm.exam        Question/Answer/Exam vocabulary, generation, grading
allm.students    Student protocol, scripted double, ModelStudent,
                 confidence parsing, versioned FailureLog,
                 StudentIdentity (mission + domain specialization)
allm.teacher     Teacher API + persistent KnowledgeState + expert lookup
allm.trainer     Trainer protocol; in-context + LoRA backends
allm.planner     need-based roadmap; mission importance scaling;
                 ingest router (concept -> specialist students)
allm.knowledge   versioned concept graph; exports the planner catalog
allm.memory      append-only episodic memory with recall + search
                 (exam engine extends allm.exam: model-generated exams,
                 coding + composite graders)
allm.debate      independent answers, clustered; disagreement signal;
                 unresolved debates -> learning tasks
allm.compression evidence-preserving abstraction with probe-guarded
                 retraction
allm.collector   deduplicating, quality-scoring sample pool
allm.evaluation  Plan.md metrics; holdout-gap diagnosis; marginal learning
                 gain per strategy phase; controlled ablation comparison
allm.kdp         Knowledge Distillation Pipeline: raw documents ->
                 knowledge units -> graph injection (see KDP.md)
allm.kel         Knowledge Evaluation Layer: six epistemic metrics,
                 Graph Health Score, failure-mode diagnosis (see KEL.md)
allm.evidence    Evidence Packages: contributions as traceable evidence;
                 replication-aware confidence (see smallVision.md)
allm.proposals   experiment proposals: AI questions -> human evidence
                 (open -> claimed -> resolved-by-evidence)
allm.api         FastAPI boundary for the platform (api extras)
allm.researcher  discover → package → recommend; model router (M46);
                 curriculum diagnostics; never teaches students
      ▲
allm.loop        composition root; optional ResearcherLayer feed to planner
```

Dependencies point downward only. A future phase may depend on core,
storage, tracking, models and data — never on the CLI, and never on a
sibling phase's concrete classes (protocols only).

## Key decisions (Phase 1)

| Decision | Rationale |
|---|---|
| Pydantic frozen models for config | validation + immutability; layered defaults < YAML < env (`ALLM_SECTION__FIELD`) < explicit overrides |
| stdlib `logging` under an `allm` namespace | universal interop; optional JSON formatter for machine-readable experiment logs |
| `Registry` + `allm.plugins` entry points | implementations are named in config, third parties can add backends without forking |
| Minimal DI container (factories + singletons) | one composition root, no framework magic to fight in a research codebase |
| SQLite append-only `records(namespace, key, version, value, reason, created_at)` | zero setup; schema trivial enough to move to PostgreSQL by copying rows |
| File-per-run local tracker (`meta.json`, `params.json`, `metrics.jsonl`, `artifacts/`) | greppable and diffable; hosted trackers are just another backend |
| `ModelSpec`/`DatasetSpec` as YAML-loadable data | teachers and students become configuration, not code |
| `EchoModel` first-class test double | the entire learning loop must be assertable in CI without model weights |

## Key decisions (Phase 2 — Teacher)

| Decision | Rationale |
|---|---|
| Exam vocabulary lives in `allm.exam` | teacher and students share types without depending on each other; `students -> exam <- teacher`, never `students <-> teacher` |
| Teacher sees only the `Student` protocol | Phase 3's model-backed students slot in without touching the teacher; students structurally *cannot* modify the teacher (they never hold a reference to it) |
| Teacher is pure orchestration | exam creation (`ExamGenerator`), scoring (`Grader`) and persistence (`KnowledgeState`) are injected; each swaps independently |
| `KnowledgeState` = typed layer over the versioned store | confidence history and "previous belief / new belief / reason" come free from Phase 1's append-only storage |
| Confidence is an EMA over topic exam scores | one tunable (`confidence_smoothing`); latest-score-wins and long-memory are both reachable by config |
| Phase 2 goals = weakness heuristic | `priority = 1 - confidence` below a threshold; Phase 4's planner replaces the scoring, not the API |
| `DatasetExamGenerator` + `ExactMatchGrader` first | honest deterministic evaluation now; generative exams and LLM judges are Phase 7 registry entries behind the same protocols |

## Key decisions (Phase 3 — Students)

| Decision | Rationale |
|---|---|
| Studying = memory-augmented generation | `ModelStudent` keeps a bounded note store; exact retrieval hits answer from memory, otherwise recent notes are injected into the model prompt. Learning is real and measurable today, independent of GPUs |
| Self-reported confidence (`CONFIDENCE: x` line) | weakest honest signal, but it fixes the interface; a log-prob estimator can be added for backends that expose probabilities without touching students |
| `FailureLog` over the versioned store, `to_sample()` back-conversion | Plan.md principle 4 made mechanical: every mistake is persisted with the student's confidence at the time and can be fed straight back into a trainer |
| `Trainer` protocol with `in_context` as first backend | weight-level fine-tuning (LoRA/peft) will be a second registry entry behind the same protocol — deliberately *not* shipped until it can be exercised against real models, per the no-placeholder rule |
| Unlabelled samples are counted as skipped, never silently dropped | training reports must be trustworthy for the learning loop to reason about what was actually studied |

## Key decisions (Phase 4 — Planner)

| Decision | Rationale |
|---|---|
| Need = weakness x importance x curiosity x novelty | straight from Plan.md's curiosity engine; every factor is visible on the `RoadmapItem` so rankings are always explainable |
| Novelty = 1 / (1 + observations); unexamined = fully weak | topics never measured are maximally novel and weak — the planner is naturally curious about the unknown |
| Blocked topics sort last; their urgency boosts prerequisites | wanting quantum gravity makes general relativity urgent; unknown prerequisites count as unmet because unmeasured knowledge cannot be assumed |
| Signals = knowledge state + topic catalog | the catalog (YAML for now) declares importance/curiosity/dependencies; Phase 5's knowledge graph becomes its generator without changing `TopicSignal` |
| `Roadmap.to_goals()` bridges planner -> teacher | goals stay the teacher's currency; zero-need items are dropped as noise |

## Key decisions (Phase 5 — Knowledge Graph)

| Decision | Rationale |
|---|---|
| Nodes are versioned records; edges live on the nodes | store stays schema-agnostic, whole graph greppable; a graph index can be added behind the same class if scale demands it |
| `revise()` is additive for edges and evidence | evidence can be appended, never removed — Plan.md's "never lose supporting evidence" is enforced structurally, which Phase 9 compression must obey |
| Mandatory `reason` on every revision | belief changes are always explainable; history = record history |
| Prerequisite cycles rejected at write time | a circular curriculum can never be scheduled; failing at write beats failing in the planner |
| `to_catalog()` bridges graph -> planner | usefulness maps to importance, prerequisites to dependencies; the hand-written YAML catalog becomes obsolete once concepts exist |

## Key decisions (Phase 6 — Memory)

| Decision | Rationale |
|---|---|
| Memory = episodic event log, not a second copy of beliefs | confidence history and revisions already live versioned in state/graph; duplicating them would create two sources of truth |
| `remember_exam` bridge | graded exams become success/failure episodes with reasoning traces mechanically — the loop never hand-crafts memory writes |
| Lexical (token-overlap) search first | dependency-free and deterministic; FAISS/Chroma vector recall is a second `memory_backends` entry once semantic recall is actually needed |
| Episode ids from a store-seeded sequence | restarts cannot silently version-over old episodes |

## Key decisions (Phase 7 — Exam Engine)

| Decision | Rationale |
|---|---|
| `ModelExamGenerator` parses a strict `T:/Q:/A:` block format | any LanguageModel can write exams; a scripted echo model makes generation fully testable offline |
| Too few questions = warning; zero questions = error | silent empty exams would corrupt every downstream confidence estimate |
| Kind + difficulty are constructor params, not mutable state | the loop raises difficulty by building a new generator; runs stay reproducible |
| `CodingGrader` executes submissions in a subprocess (`-I`, hard timeout) | scoring code by string match would be meaningless; caveat documented — needs OS-level sandboxing before grading untrusted model output |
| `CompositeGrader` routes by `question.kind` | the teacher keeps exactly one grader while mixed exams still grade correctly per question |
| Weakness storage unchanged | teacher state + FailureLog from Phases 2-3 already fulfil "store weaknesses" |

## Key decisions (Phase 8 — Debate)

| Decision | Rationale |
|---|---|
| Students answer independently (no cross-talk yet) | the first thing the system needs is an *honest disagreement signal*; persuasion dynamics would contaminate it and can be layered on later |
| Disagreement = 1 - largest cluster / participants | simple, bounded, explainable |
| Verdict by total cluster confidence, not head-count | two sure students outweigh three shruggers; with ground truth each position is also graded so confident-but-wrong majorities are visible to the teacher |
| Unresolved debates -> `to_learning_sample()` | with ground truth it is trainable; without it the target is `None` — an open research task, exactly Plan.md's "disagreements become new research tasks" |

## Key decisions (Phase 9 — Compression)

| Decision | Rationale |
|---|---|
| Compression = additive abstraction, never deletion | the principle concept carries the union of member evidence; members stay in the graph linked to it — "never lose supporting evidence" holds even when compressing |
| Candidates = identical non-empty prerequisite sets | several ideas resting on exactly the same foundations are the classic shape of a hidden common principle; richer similarity can extend `propose()` later |
| Probe-guarded application with *retraction* | if the injected `PerformanceProbe` reports regression beyond tolerance, the principle's status flips to `retracted` with the scores in the reason — history intact, catalog cleaned |
| Principle confidence = min of members | an abstraction is only as trustworthy as its weakest support |

## Key decisions (Phase 10 — Collector, Evaluation, Loop)

| Decision | Rationale |
|---|---|
| `SamplePool` dedupes by normalised input; labelled upgrades unlabelled | any source (datasets, failure logs, debate outcomes) feeds one pool; web/paper acquisition later is just another source |
| Evaluation metrics are derived, never stored | improvement, learning speed, mastery and self-correction are computed from teacher state + memory on demand, so they cannot drift from ground truth |
| Self-correction matches questions by *prompt* | question ids are exam-specific and never recur; `None` (no failures yet) is distinct from 0.0 (never corrects) |
| The loop is a pure composition root | it owns the *order* of Plan.md's stages, zero logic of its own; debate, graph and compression are optional injections |
| Measure and test use different exams within an iteration | the per-iteration delta measures learning, not memorisation of the measure exam |
| The loop keeps no private state | exams, failures, goals, episodes and metrics are persisted as they happen; a crash loses nothing |

## Key decisions (M4 — KEL-steered loop & learning strategies)

| Decision | Rationale |
|---|---|
| `LearningStrategy` phases: definitions → relations → reasoning → research | curriculum depth before breadth; each phase changes sample kinds, paraphrase and failure-study behaviour |
| KEL steers strategy advance on peak/window held-out score, not last score alone | exam sampling is noisy; peak over a window avoids advancing or halting on one lucky draw |
| Static-illusion halt requires min iterations + min LG history + no improving peak | GST=1.0 on a tiny early graph is misleading; premature halt blocked the strategy timeline |
| `learning_history.jsonl` per iteration | research dataset: strategy, samples, scores, KEL LG, failure prompts — enables marginal gain analysis |
| Marginal learning gain (MLG) per strategy phase | answers which phase actually paid off, not just which came before a spike |
| `ALLM_LOOP_SEED` fixes exam draws across ablation arms | controlled experiments isolate one variable; same seed → same questions |

## Key decisions (Specialist students)

| Decision | Rationale |
|---|---|
| `StudentIdentity` YAML: mission, core/primary/secondary/ignored domains, exploration rate | not every student learns everything; shared core (~15%) + specialization (~80%) + 5% exploration |
| `domain_fit()` scales planner importance; zero outside mission | KEL/planner question becomes "does this help my mission?" — ignored domains get zero need |
| `IngestRouter` assigns KDP concepts to matching specialists | incoming documents route by topic overlap, not broadcast to all students |
| `rank_experts` / `best_expert` on `KnowledgeState` | peer consultation and debate can invite the right specialist |
| Mission filter on collected samples in `LearningLoop` | specialists only study in-mission material, not the full pool |
| Mixed corpus (`kdp/mixed_corpus.py`) | plasma workshops + software fixture for honest multi-domain experiments |
| Dual-specialist ablation (`examples/23`) | generalist vs routed specialists on domain-specific held-out exams |
| `request_consultation` | peer expert lookup when a student needs cross-domain help |
| `consultation_samples` + `enable_peer_consultation` | failed out-of-mission exam topics pull pool samples via expert routing |
| Identity ablation (`examples/22`) | one-knob experiment: mission on vs off, same seed, export `ablation_comparison.json` |

## Key decisions (Researcher — distributed acquisition)

| Decision | Rationale |
|---|---|
| Researcher never trains students or assigns them | one curriculum path: external → package → Teacher → students |
| `KnowledgePackage` is the exchange format | concepts, definitions, evidence, conflicts, provenance — handoff to Teacher/KDP |
| `ResearchRecommendation` boosts planner importance | Researcher recommends; Teacher/KEL routes via planner + ingest router |
| Providers registered (`workshop`, `software`) | v0: Kids MK + software fixture; GitHub/papers are future providers |
| Contradictions preserved in packages | disagreement is first-class, never deleted at packaging |
| `RecommendationQueue` is append-only | same storage invariants as the rest of ALLM |
| Teacher-mediated consultation | specialist explains → Teacher grades → only approved material reaches asker |

## Key decisions (KDP — Knowledge Distillation Pipeline)

| Decision | Rationale |
|---|---|
| Fully deterministic, rule-based stages (no LLM, no randomness) | KDP.md 7.5 mandates same input -> same output; even ids are content hashes. Embedding clustering (Stage 5) is a planned upgrade and must use pinned models to keep the guarantee |
| Raw documents persisted untouched, spans paragraph-granular | provenance-first (7.3): every unit traces to raw text; character-precise spans inside cleaned paragraphs are future work, the paragraph span is always exact |
| Cleaning list is conservative | dropping "um/uh/you know" is safe; words like "like"/"well" carry meaning too often — keeping noise beats losing semantics (7.1) |
| Extraction is pattern-based; unmatched segments yield nothing | that is the noise filter, not data loss (raw text survives in Stage 1); quality improves by adding patterns, never by loosening provenance |
| Concept-name quality is bounded by rule-based subject detection | crude hints ("To Compute It First") are a known, documented limitation the alias table and future embedding stage address |
| Merging is additive; conflicts are first-class | perspectives concatenated, sources unioned, divergent definitions become `ConflictNode`s that penalise confidence and land in the store for exams/debate |
| Confidence = 0.4·frequency + 0.3·consistency + 0.3·clarity, halved under conflict | stability, not truth (KDP.md 5); fixed weights keep scores reproducible and explainable |
| Injection adds or *revises*, never resets confidence of existing concepts | KDP measures textual stability; mastery confidence belongs to the teacher's exam loop |

## Key decisions (KEL — Knowledge Evaluation Layer)

| Decision | Rationale |
|---|---|
| Metrics are pure functions over already-recorded data | KEL.md sections 1+4: measurement-only, no external evaluators; a metric that needs its own bookkeeping can drift from the truth it claims to measure |
| `None` = "cannot measure", never 0.0 | a system with no conflicts has *unmeasurable* CRE, not perfect CRE; conflating the two would fake trends |
| GHS uses the spec's weights verbatim; computed only when every component exists | a composite over missing data breaks comparability across time — the first evaluation deliberately has no GHS |
| CRR enters GHS as `min(1, crr/target_reuse)`; LG enters raw | the spec leaves normalisation open: CRR is unbounded so it needs a target; LG in [-1,1] stays raw so negative learning genuinely drags the score down |
| GST = Jaccard over a structural fingerprint (active nodes + edges) | cheap, deterministic, and blind to prose edits — stability should measure structure, not wording |
| KEL's own measurements are versioned records (`kel_metrics`, `kel_snapshots`) | time series come free from the append-only store, and measurements about the system get the same never-overwrite treatment as knowledge itself |
| Failure modes read the *latest recorded* metrics, not a fresh evaluation | diagnosis must be reproducible against what was actually measured, and testable by injecting synthetic series |

## Key decisions (Evidence — smallVision.md)

| Decision | Rationale |
|---|---|
| Packages are immutable, content-addressed, blob-free | the platform stores files; the core stores claims, references (URI + sha256) and outcomes — chain- and platform-agnostic |
| Per-(contributor, stance) only the strongest package counts | popularity resistance: 50 posts by one lab move confidence exactly as much as 1; a second independent contributor moves it a lot |
| Replication weight only for re-running *someone else's* package | self-replication is just another experiment; independence is the thing being rewarded |
| Confidence = Laplace-smoothed support share; inconclusive enters the denominator at half weight | one unchallenged experiment lands near 0.6, not 1.0 — certainty is earned; inconclusive results are real uncertainty, not noise |
| Every confidence value ships with a `ConfidenceBreakdown` | "nothing is hidden": weights, contributor count, replications and every package id behind the number |
| Binder recomputes concept confidence from the full ledger on every submit | belief is always a pure function of the evidence; nobody sets confidence by hand |
| Three confidence types stay separate | textual stability (KDP), evidential confidence (packages — the graph's primary belief signal), student mastery (teacher state) — blending them silently would fake epistemic progress |

## Key decisions (Proposals — smallVision.md)

| Decision | Rationale |
|---|---|
| Proposals resolve only through evidence packages | nobody settles a question by decree; the packages' evidential confidence decides supported/challenged/inconclusive |
| Factories from debates, KDP conflicts and roadmap items | the "AI suggests what to test next" arrow of the vision is mechanical, not manual |
| Duplicate (concept, question) proposals collapse while unresolved | humans never see the same ask twice |
| Every transition is a new versioned record with a reason | the negotiation history (who claimed, what resolved it) is part of the knowledge |
| KDP Stage 7 also detects numeric contradictions | definitions sharing vocabulary but disagreeing on quantities ("exceeds 80%" vs "capped at 10%") are the classic experimental dispute — surfaced while building the evidence demo |

## Key decisions (API — platform boundary)

| Decision | Rationale |
|---|---|
| One factory, one SQLite path | `create_app(db)` wires the entire core; deployment for the platform is one file + one process |
| Identity, incentives, file storage stay outside | the core sees opaque contributor ids and artifact URIs — chain-agnostic by construction, SocialFi mechanics live in the platform |
| `POST /documents` auto-opens proposals for detected conflicts | "AI suggests what to test next" happens at ingestion, not when an operator remembers to ask |
| Wire schemas are separate from domain types | the HTTP contract can freeze or evolve independently of internal models |
| KEL evaluation is a POST | taking a measurement appends to the time series; mutating GETs make trends lie |
| Invalid lifecycle transitions map to 409 | the platform can retry/inform users; nothing crashes the core |

## Key decisions (Researcher — ResearcherPlan.md)

| Decision | Rationale |
|---|---|
| Researcher recommends; Teacher owns curriculum | students never ingest raw external sources; KEL + planner merge recommendations into importance/curiosity |
| `KnowledgePackage` is the exchange format | language-independent bundles with provenance, conflicts preserved for debate/proposals |
| `Provider` registry with offline fixtures first | workshops + software JSONL exercise the full pipeline before live crawlers or federation |
| Lazy exports in `allm.researcher` | avoids import cycles with KDP → knowledge graph → planner |
| `curriculum_topic` + `align_recommendation_topic` | KDP fragment labels map to pool topics (`kids-plasma`); `studied>0` in loop |
| `merge_research_recommendations` lives outside `planner.__init__` | callers import from `allm.planner.researcher_signals` so the planner package loads without pulling researcher/KDP |
| Teacher-mediated consultation | specialist explains → Teacher grades → only approved samples enter the asker's study queue |
| `LearningLoop(researcher=…)` optional feed | Researcher cycle runs once (or on schedule); active recommendations boost catalog each iteration |
| Capability pipeline L0–L7 | Composable skills with append-only metrics; see `ECOSYSTEM_CAPABILITIES.md` |
| `ALLM_RESEARCHER_TARGETING=1` | Per-student recommendation filter in learning loop |

## Key decisions (M45 — KEL research requests)

| Decision | Rationale |
|---|---|
| KEL emits structured `KelResearchRequest` on repair/stagnation/conflict/gap | closes the loop: measurement triggers investigation, not just logging |
| Researcher enqueues `remediation` recommendations | Teacher/planner receive actionable tasks, not raw findings |
| Requests persisted in store (`kel_research_requests`) | next Researcher cycle and diagnostics can read pending work |
| Triggers: repair_mode, strategy_stagnation, unstable_mastery, high_conflict, research_gap, forgetting | each maps to a different teaching failure mode |

## Key decisions (M46 — Chief Scientist orchestrator)

| Decision | Rationale |
|---|---|
| Researcher routes by task, not one monolithic model | reasoning / verifier / vision specialists — expensive work once, many cheap students |
| `ResearcherModelRouter` maps KEL triggers + hints → specialist role | conflict → verifier; relations/prerequisites → reasoning; visual hints → vision |
| `CurriculumDiagnostic` = failure_reason + confidence + evidence + recommendations | answers *why* relations fail, not just "add more definitions" |
| Heuristic diagnostics always available; reasoning model optional (`auto`) | fast CI/offline runs; 14B Chief Scientist when Ollama reachable |
| Diagnostics run on KEL submit + `diagnostics.curriculum` capability | mid-loop enrichment of remediation proposals with `Diagnosis: …` |
| Learning history persisted to store per iteration | failure prompts and strategy phases feed prerequisite analysis |

### Model roles in the current Ollama stack

| Role | Default model | Responsibility |
|------|---------------|----------------|
| Student | `qwen2.5:7b-instruct` | Answer exams; learn from Teacher-delivered material |
| Grader / Judge | `qwen2.5:14b-instruct` | Curriculum + alignment + evidence grading (M44) |
| Teacher | *(no LLM)* | Exams, goals, visual approval, curriculum orchestration |
| Researcher reasoning | `qwen2.5:14b-instruct` | Curriculum diagnostics, remediation planning |
| Researcher verifier | `qwen2.5:14b-instruct` | Conflict triage, cross-source reconciliation |
| Researcher vision | `llava` | Book figure caption + OCR (M13/M27) |

Only the **Researcher** may introduce new knowledge into the ecosystem (after
discovery + verification). Teachers organize and approve; students learn.

## Researcher capability pipeline

```
observe.curiosity → analysis.gap → missions.review → planning.research
  → discovery (workshop | book | video | livekit | software)
  → understanding.package → understanding.book.images
  → understanding.sync → understanding.vision → understanding.ocr
  → understanding.visual.distill → verification.graph
  → diagnostics.curriculum (M46) → curriculum.target (L4)
  → ecosystem.analyze → economy.ledger → improvement.reflect (L7)
        │
        ▼
KnowledgePackage + ResearchRecommendation → Teacher / KEL
```

### Closed learning loop (M45+)

```
Student → Exam → KEL (multi-objective) → KelResearchRequest
    → Researcher (diagnostics + remediation) → Teacher → Student
```

KEL research requests are submitted each iteration when compromise mode is
`repair`/`maintain`, strategy stagnates, or findings flag unstable mastery,
high conflict, or research gaps.

The original L0–L7 labels (plan → discover → understand → verify → curriculum →
ecosystem → economy → improve) still apply; multimodal book/vision capabilities
and M46 diagnostics extend the pipeline without replacing it.

## What comes next (proposed, not yet implemented)

- **Remediation execution** — Researcher diagnostics today enrich recommendations;
  automatic package generation from diagnostic hypotheses is not yet wired.
- **Live discovery providers** — GitHub, papers, docs (protocol slots exist; offline fixtures only today).
- **Log-prob confidence estimator** — for model backends that expose
  token probabilities; replaces self-reported confidence where possible.
- **Vector memory backend** (FAISS/Chroma) — semantic recall behind the
  existing `memory_backends` registry.
- **OS-level sandboxing for `CodingGrader`** — required before grading
  untrusted model output.
- **Debate with argument exchange** — persuasion rounds on top of the
  existing disagreement signal and result types.
- **Autonomous data collection** — web/book/paper sources feeding the
  same `SamplePool` contract, with provenance-aware quality scoring.
- **KDP embedding stage at scale** — sentence-transformers clustering on
  full Kids corpus; character-precise span mapping through cleaning.
- **LoRA under fixed curriculum** — held-out transfer with specialist
  identity held constant; pause scale-up until in-context transfer stabilises.
