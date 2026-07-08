# RESEARCHER.md — Distributed Knowledge Acquisition Layer

See also [`ResearcherPlan.md`](../../ResearcherPlan.md) for the full vision.

## Purpose

The Researcher decides **what the ecosystem should learn next**. It discovers
external knowledge, evaluates sources, builds Knowledge Packages, detects
contradictions, and **recommends** topics to the Teacher — it never teaches
students directly.

```
External World → Researcher → Knowledge Packages → Teacher (KEL) → Students
```

## Module: `allm.researcher`

| Component | Role |
|-----------|------|
| `KnowledgePackage` | Language-independent exchange format (concepts, evidence, provenance) |
| `Provider` | Registered source (workshops, software fixture, future: GitHub, papers) |
| `ResearcherLayer` | discover → evaluate → package → recommend |
| `RecommendationQueue` | Append-only recommendations for the Teacher/planner |
| `merge_research_recommendations` | Boost planner importance from Researcher priorities |
| `curriculum_topic` on `KnowledgePackage` | Workshop packages declare pool topic (e.g. `kids-plasma`) |
| `align_recommendation_topic` | Maps noisy KDP labels to catalog / pool topics |
| `ResearchRecommendation.concept` | Fine-grained source concept when topic is aligned |
| `ResearcherEcosystemMetrics` | KEL feed: missing knowledge, saturation, conflicts, growth |
| `CapabilityPipeline` | L0–L7 orchestration; see [`ECOSYSTEM_CAPABILITIES.md`](ECOSYSTEM_CAPABILITIES.md) |
| `ResearchMission` / `MissionStore` | Persistent research goals above per-cycle plans |
| `CuriositySignal` | Proactive questions ranked before planning |
| `KnowledgeTier` | `established` / `emerging` / `hypothesis` on concepts and recommendations |
| `ResearcherReport` | Adds `curiosity_signals`, `graph_gaps`, `active_missions`, `multimodal_synced` |
| `SyncedEvidence` | Timestamp-aligned transcript + visual/audio cues on packages |
| `EvidenceBroker` | Search packages for debate/Teacher "show me" requests |

## Capability layers

See [`ECOSYSTEM_CAPABILITIES.md`](ECOSYSTEM_CAPABILITIES.md) for the full skill-to-module map.  
Decision cycle and philosophy: [`RESEARCHER_PHILOSOPHY.md`](RESEARCHER_PHILOSOPHY.md).

## Invariants

1. Researcher never calls `Trainer.train` or assigns students.
2. Every package has provenance and a provider id.
3. Contradictions are preserved, not deleted.
4. Recommendations are suggestions; Teacher/KEL owns curriculum.

## Non-goals (v0)

- Live web crawling
- Cross-network ALLM federation
- Automatic provider reputation from citations (stub only)
