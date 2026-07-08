# KDP.md — Knowledge Distillation Pipeline (ALLM Module)

## Purpose

The Knowledge Distillation Pipeline (KDP) converts **raw, noisy human knowledge sources** (e.g. transcripts, PDFs, lectures) into **structured, deduplicated, machine-learnable knowledge units** that can be consumed by ALLM Phase 5 (Knowledge Graph).

KDP is not a model.

It is a **deterministic transformation system**.

---

# 1. Position in ALLM

KDP sits between:

```text id="kdp-flow-1"
Raw Sources → KDP → Knowledge Graph (Phase 5) → Planner/Teacher/Exam/Memory
```

It is the **only entry point for unstructured knowledge**.

---

# 2. Input Types

KDP accepts:

* workshop transcripts
* PDFs (converted to text)
* lecture recordings (transcribed)
* notes / markdown
* Q&A logs
* mixed web exports

All inputs are treated equally:

> “human explanation streams”

---

# 3. Output Types

KDP produces **Knowledge Units (KU)**:

```json id="kdp-ku-1"
{
  "id": "ku_self_attention_001",
  "type": "concept | fact | procedure | misconception | question",
  "content": "...",
  "normalized_concept": "Self-Attention",
  "confidence": 0.0,
  "sources": ["doc_12", "doc_87"],
  "context": "transformers",
  "relations": [],
  "tags": ["nlp", "deep_learning"],
  "raw_span_refs": [
    {"doc": "doc_12", "start": 102, "end": 280}
  ]
}
```

---

# 4. Pipeline Overview

KDP is a **multi-stage deterministic compiler**.

## Stage 1 — Ingestion

* load documents
* assign IDs
* preserve raw text (never modified)

Output:

```text id="kdp-s1"
DocumentStore[]
```

---

## Stage 2 — Cleaning

Remove:

* filler speech (“um”, “you know”)
* repetitions
* formatting noise
* irrelevant side talk

BUT:

> never delete semantic meaning

Output:

```text id="kdp-s2"
CleanSegments[]
```

---

## Stage 3 — Segmentation

Split into **semantic units**, not fixed length.

Rules:

* one idea per segment
* split on topic shifts
* preserve context links

Output:

```text id="kdp-s3"
Segments[]
```

---

## Stage 4 — Concept Extraction

Extract:

* definitions
* claims
* procedures
* examples
* misconceptions
* implicit assumptions

Output:

```text id="kdp-s4"
RawKnowledgeUnits[]
```

---

## Stage 5 — Concept Normalization

Merge semantically similar units.

Rules:

* embedding similarity clustering
* alias detection
* phrase normalization

Example:

* “self attention mechanism”
* “attention between tokens”

→

> Self-Attention

Output:

```text id="kdp-s5"
NormalizedUnits[]
```

---

## Stage 6 — Deduplication + Evidence Merge

For identical concepts:

* merge content
* preserve all explanations
* attach all sources
* keep multiple perspectives

Rule:

> NEVER overwrite knowledge — only expand it

---

## Stage 7 — Contradiction Detection

If two units disagree:

Create:

```text id="kdp-s7"
ConflictNode {
  concept: "X",
  interpretation_a: "...",
  interpretation_b: "...",
  evidence: [...]
}
```

Conflicts are NOT errors.

They are:

* training signals
* exam material
* reasoning tasks

---

## Stage 8 — Knowledge Unit Finalization

Each KU is finalized with:

* normalized concept name
* structured content
* confidence score
* relations (optional)
* source mapping

Output:

```text id="kdp-s8"
KnowledgeUnits[]
```

---

## Stage 9 — Graph Injection (Phase 5)

KDP output is written into:

> ALLM Knowledge Graph

Rules:

* append-only
* versioned
* provenance preserved
* no deletion

---

# 5. Confidence Model

Confidence is computed from:

* frequency across sources
* consistency of definitions
* absence of contradiction
* clarity of structure
* stability over time

Important:

> confidence is NOT truth — it is stability

---

# 6. Deduplication Philosophy

KDP does NOT try to find “the correct version”.

It tries to build:

> the most structurally complete representation of a concept

So it prefers:

* merging perspectives
* preserving disagreement
* increasing evidence coverage

---

# 7. Key Design Principles

## 7.1 Lossless Knowledge Compression

No semantic loss allowed.

## 7.2 Atomicity

Each KU = one idea.

## 7.3 Provenance First

Every KU must trace back to raw source spans.

## 7.4 Conflict Preservation

Contradictions are first-class outputs.

## 7.5 Determinism (important)

Same input → same output.

---

# 8. Non-Goals

KDP does NOT:

* decide curriculum priorities
* teach students directly
* optimize learning strategy
* rewrite knowledge into “clean explanations”

It ONLY transforms raw data → structured knowledge.

---

# 9. Integration with ALLM Phases

## Phase 5 (Knowledge Graph)

KDP is the primary ingestion system.

## Phase 4 (Planner)

Uses normalized concepts + dependencies.

## Phase 2 (Teacher)

Uses KDP-generated knowledge for exams.

## Phase 7 (Exam Engine)

Generates questions from KUs + conflicts.

## Phase 8 (Debate)

Uses contradictions from KDP.

## Phase 9 (Compression)

Uses high-confidence merged units.

---

# 10. Success Criteria

KDP is correct if:

* 600 transcripts collapse into a few thousand atomic units
* repeated explanations become unified concepts
* contradictions are explicitly visible
* graph becomes traversable curriculum structure
* planner can rank learning priorities without raw text

---

# 11. Final Principle

> KDP is not about understanding knowledge.
> It is about making knowledge structurally usable.
