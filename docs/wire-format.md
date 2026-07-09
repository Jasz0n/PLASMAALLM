# The ALLM wire format

*The public contract platform teams build against — version 1.0.0.*

You should not have to read ALLM's source, or track its package version,
to integrate with it. This document and its machine-readable companion
[`wire-format.json`](wire-format.json) are the whole contract: what a
contributor sends, what the core sends back, and what the event feed
streams. Fetch the live copy from a running instance at `GET /wire`, or
export it with `allm wire -o wire-format.json`.

The schemas in `wire-format.json` are **generated from the running
models**, and a CI drift guard fails the build if the published file and
the implementation disagree — so "documented" and "implemented" cannot
diverge.

## Versioning promise

`wire_version` (currently **1.0.0**) is the contract's own semantic
version, deliberately **separate from the engine version**
(`generated_by`). The engine can be refactored freely; the contract only
moves when *your* integration would notice:

| Change | Version bump | You must… |
|---|---|---|
| New **optional** request field; new response field; new event `type`; new `PackageKind`/`Outcome` value | **minor** (1.x) | nothing — stay forward-compatible by ignoring unknown fields |
| A required field added or removed; a type narrowed; a vocabulary value retired; a field's meaning changed | **major** (2.0) | update before upgrading |

Two rules make minor bumps safe to ignore: **tolerant reader** — ignore
response fields you don't recognise; **don't over-validate vocabularies**
— treat `kind`/`outcome` as open strings you map, not a closed enum you
reject on.

## What you send

- **`EvidenceSubmission`** → `POST /evidence`. The heart of the contract:
  a `claim`, the `concept` it bears on, a `contributor` id (opaque to the
  core — identity is the platform's), an `outcome`
  (`supported` / `challenged` / `inconclusive`), and a `kind`. Artifacts
  are **references, never blobs** — the platform stores files; the core
  stores their `uri` + `sha256`. `replicates` names the package this one
  independently re-runs: replication is what actually moves confidence.
  Size caps are generous for honest use and hostile to payload abuse.
- **`DocumentSubmission`** → `POST /documents`. One raw human explanation
  stream; the core distills it and auto-opens proposals for conflicts.
- **`ClaimRequest`** / **`ResolveRequest`** → the proposal lifecycle
  (`/proposals/{id}/claim`, `/proposals/{id}/resolve`).

## What you get back

- **`ConfidenceBreakdown`** — returned with every confidence value so
  nothing is hidden: the `value`, the support/challenge/inconclusive
  weights behind it, the `contributors` and `independent_replications`
  counts, and every `packages` id that entered the number. **Popularity
  cannot move belief** — only independent replication does.
- **`ConceptSummary`** — the list-view shape from `GET /concepts`.
- **`Event`** — one entry of the live feed (`GET /events?since=<seq>`).
  Read it three ways, same `seq` semantics throughout: poll
  `GET /events?since=<seq>`, subscribe to the **Server-Sent Events**
  stream `GET /events/stream` (a browser `EventSource` that auto-resumes
  from `Last-Event-ID`), or register a webhook (below).
  Each carries a monotonic `seq`; poll with the last `seq` you saw and
  you will never miss or replay one. Types today: `evidence.submitted`,
  `confidence.changed`, `proposal.opened`, `proposal.resolved`,
  `contribution.proposed` / `approved` / `rejected` / `applied`, and
  `workshop.observed`. Treat the type set as open (new ones are a minor
  bump) and ignore any you don't handle.

## Receiving webhooks

Instead of polling `GET /events`, a platform can register an endpoint to
have the feed **pushed** to it. Delivery is opt-in and approval-gated:
`POST /webhooks` registers a *proposed* subscription (returning a
`secret` **once** — store it), and it delivers nothing until a named
human calls `POST /webhooks/{id}/approve`. Every attempt is recorded and
readable at `GET /webhooks/deliveries`.

Each delivery is a `POST` to your URL with this body:

```json
{ "wire_version": "1.0.0", "event": { "seq": 42, "type": "confidence.changed", "subject": "plasma", "data": {"value": 0.81}, "created_at": "…" } }
```

and these headers:

| Header | Meaning |
|---|---|
| `X-ALLM-Event` | the event `type` |
| `X-ALLM-Event-Seq` | the monotonic `seq` (dedupe / order on this) |
| `X-ALLM-Signature` | `sha256=` + HMAC-SHA256 of the **raw body** keyed by your `secret` |

**Verify every delivery**: recompute the HMAC over the exact bytes you
received and constant-time-compare it to `X-ALLM-Signature`. Return any
`2xx` to acknowledge; a non-2xx or a timeout is recorded as a failed
delivery. Treat delivery as at-least-once and possibly out-of-order —
`seq` is your dedupe and ordering key.

## What stays outside the contract, on purpose

Identity, rewards and file storage are the platform's. The core sees
opaque `contributor` ids and artifact `uri`s and never mints, ranks or
pays them. This is not a gap — it is the boundary that lets the
replication-aware confidence model stay the anti-gaming primitive.
