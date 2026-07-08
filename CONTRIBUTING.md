# Contributing to ALLM

ALLM is an experimental research platform for autonomous learning in
language models, and the knowledge engine behind an open evidence-based
research platform. Contributions of every size are welcome — code,
specs, evidence-model critique, and replication of our own claims.

## Ground rules (the invariants)

These are architectural laws, not preferences. PRs that violate them
will be asked to change:

1. **Nothing is ever overwritten.** All persistent state goes through
   the versioned record store with a `reason` per write.
2. **Everything is replaceable.** Components depend on `Protocol`
   interfaces; concrete implementations register by name and are chosen
   via configuration.
3. **Everything runs offline.** Heavy dependencies (torch, transformers,
   fastapi) are optional extras behind lazy imports; every feature must
   be testable with the deterministic `echo` model and temp storage.
4. **Confidence is earned, never set.** Evidential confidence is a pure
   function of evidence packages; textual stability (KDP), evidential
   confidence (graph), and student mastery (teacher) are never blended.
5. **No placeholder code.** Ship a smaller real thing rather than a
   larger fake one.

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # core + tests
pip install -e ".[api]"      # optional: HTTP API
pytest                        # must stay green, offline, in seconds
```

## Making changes

- Read `Plan.md` (vision), `docs/architecture.md` (decisions), and the
  module spec (`KDP.md`, `KEL.md`, ...) touching your area.
- Every module change ships with: typed models, tests, and — for new
  behaviour — a runnable example under `examples/`.
- Record significant design decisions as a row in the relevant
  decision table in `docs/architecture.md` (decision + rationale).
- Keep files small and single-responsibility; prefer composition.

## Proposing new modules

Larger ideas follow the spec-first flow this repo is built on: write a
`<MODULE>.md` spec (purpose, position in ALLM, inputs/outputs, design
principles, non-goals), open it for discussion, then implement it as
`src/allm/<module>/` once the spec settles.
