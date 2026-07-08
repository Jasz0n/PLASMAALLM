# KEL.md — Knowledge Evaluation Layer (ALLM Module)

## Purpose

The Knowledge Evaluation Layer (KEL) defines **how to measure whether the Knowledge Graph (Phase 5 + KDP output) is actually improving over time**.

It answers a critical question:

> Is ALLM learning, or only reorganizing information?

KEL is the **truth-checking layer for epistemic improvement**, not model performance.

---

# 1. Scope

KEL evaluates:

* Knowledge Graph (Phase 5)
* KDP output (Phase 5 input pipeline)
* Planner outputs (Phase 4 signals)
* Exam performance (Phase 7)
* Student learning trajectories (Phase 3)

KEL does NOT modify knowledge.

It only measures it.

---

# 2. Core Principle

> If knowledge cannot be measured, it cannot be improved.

KEL enforces:

* comparability across time
* stability of metrics
* independence from model internals

---

# 3. Evaluation Dimensions

KEL defines 7 primary metrics.

---

## 3.1 Redundancy Collapse Ratio (RCR)

Measures how well KDP compresses repeated knowledge.

```text id="kel-rcr"
RCR = 1 - (unique_concepts / total_raw_concepts)
```

Interpretation:

* high RCR → strong deduplication
* low RCR → weak compression

Goal:

> increase structure without losing meaning

---

## 3.2 Conflict Density (CD)

Measures epistemic disagreement in the graph.

```text id="kel-cd"
CD = number_of_conflict_nodes / total_concepts
```

Interpretation:

* high CD → unstable knowledge area
* low CD → highly agreed knowledge

Important:

> CD is NOT bad. It is a learning signal.

---

## 3.3 Graph Stability Over Time (GST)

Measures how much the graph changes between versions.

```text id="kel-gst"
GST = similarity(graph_t, graph_t-1)
```

Interpretation:

* high stability → mature knowledge
* low stability → active learning phase

Key insight:

> instability is expected early, dangerous late

---

## 3.4 Concept Reuse Rate (CRR)

Measures whether concepts are actually useful.

```text id="kel-crr"
CRR = avg number of downstream uses per concept
```

Used in:

* exams
* planner decisions
* student tasks

Interpretation:

* high CRR → useful abstraction
* low CRR → dead or redundant concept

---

## 3.5 Learning Gain (LG)

Measures student improvement using graph-derived tasks.

```text id="kel-lg"
LG = score_after - score_before (per concept cluster)
```

This is the **only direct learning metric**.

Important:

> This is the closest proxy to “system intelligence improvement”

---

## 3.6 Contradiction Resolution Efficiency (CRE)

Measures whether conflicts become productive learning signals.

```text id="kel-cre"
CRE = % of conflicts that lead to:
    - exam questions
    - resolved understanding
    - or refined concepts
```

Interpretation:

* high CRE → system turns disagreement into learning
* low CRE → conflicts are noise

---

## 3.7 Evidence Growth Rate (EGR)

Measures whether the **ecosystem produces better-founded knowledge**,
not just better exam scores.

Each concept's *evidence foundation* is scored from its packages:

```text id="kel-egr-foundation"
foundation(concept) =
    packages
  + distinct package kinds        (diversity: experiment ≠ paper ≠ video)
  + distinct contributors
  + 2 × independent replications  (the strongest signal)
```

EGR is the relative growth of the total foundation between
measurements:

```text id="kel-egr"
EGR = (foundation_t - foundation_t-1) / max(foundation_t-1, 1)
```

Interpretation:

* EGR > 0 → the ecosystem is earning evidence (experiments, replications, diverse sources)
* EGR = 0 → knowledge may be growing, but nothing new is *founded*
* None → no ledger connected, or first measurement (cannot measure ≠ zero)

Key insight:

> A student can improve while the knowledge base stays hearsay.
> EGR catches the difference: did the *knowledge itself* get better?

EGR is reported and tracked as a time series but deliberately **not**
part of GHS (section 6) — the composite's semantics stay comparable
across historical measurements.

---

# 4. Evaluation Sources

KEL uses ONLY derived data from:

* Knowledge Graph (Phase 5)
* KDP outputs
* Exam results (Phase 7)
* Memory logs (Phase 6)
* Planner decisions (Phase 4)

No external evaluation models required.

---

# 5. Time-Based Evaluation Model

KEL evaluates over windows:

* short-term (iterations)
* mid-term (phase cycles)
* long-term (system evolution)

Each metric is tracked as a **time series**:

```text id="kel-time"
metric(t) → metric(t+1) → metric(t+2)
```

This enables:

> trend-based intelligence assessment

---

# 6. Graph Health Score (GHS)

A single composite metric:

```text id="kel-ghs"
GHS =
  (0.25 × RCR)
+ (0.15 × CRR)
+ (0.20 × LG)
+ (0.15 × CRE)
+ (0.15 × GST_normalized)
- (0.10 × CD)
```

Interpretation:

* high GHS → system is learning effectively
* low GHS → system is reorganizing but not improving

---

# 7. Key Design Principle

KEL does NOT measure:

* model accuracy
* language fluency
* generation quality

It measures:

> structural epistemic improvement

---

# 8. Critical Insight

A system can:

* increase knowledge size
* increase graph complexity
* increase structure quality

and still:

> NOT improve intelligence

KEL exists to detect this failure mode.

---

# 9. Failure Modes Detected by KEL

## 9.1 False Compression

Many concepts merged incorrectly → high RCR but low CRR

## 9.2 Dead Knowledge Growth

Graph grows but is never used in exams → low CRR

## 9.3 Conflict Accumulation Without Resolution

High CD but low CRE

## 9.4 Static Illusion

High GST stability but declining LG

## 9.5 Unearned Confidence

A concept holds high confidence with **no evidence packages** behind
it. The ecosystem principle is *documents propose, evidence disposes*:
claims from documents enter as hypotheses; only reproducible evidence
may push confidence high. Detected when confidence exceeds the
configured cap while the ledger holds zero packages for the concept.

---

# 10. Integration with ALLM

## Phase 4 — Planner

Uses KEL to prioritize:

* unstable high-value areas

## Phase 5 — Knowledge Graph

Uses KEL to:

* detect bad merges
* identify dead nodes

## Phase 7 — Exams

Uses KEL to:

* focus on high-impact concepts

## Phase 10 — Loop

Uses KEL to:

* decide system iteration direction

---

# 11. Minimum Success Condition

The system is considered improving if:

* LG increases over time
* CRR increases without RCR collapse errors
* CRE stays high
* GST stabilizes gradually (not prematurely)
* GHS shows upward trend

---

# 12. Final Principle

> A knowledge system is not defined by how much it stores,
> but by how measurably it improves understanding over time.
