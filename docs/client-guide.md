# Build a client for ALLM

Everything you need to drive the contributor loop over HTTP, without
reading engine source. The machine-readable schema for every request and
response is [`wire-format.json`](wire-format.json) (`GET /wire` on a live
instance); this guide is the how-to on top of it.

## The shape of the API

- **Base URL** — wherever it's deployed (`http://localhost:8000` in
  [dev](deploy.md)).
- **Auth** — **reads are open; writes need a bearer token** the operator
  issues: `Authorization: Bearer <token>`. Per-principal rate limits
  apply (HTTP 429 when exceeded).
- **Errors** — every failure is the same JSON envelope, so you can handle
  them in one place:

  ```json
  { "error": { "status": 422, "message": "request validation failed",
               "type": "validation_error", "fields": [ … ] },
    "detail": … }
  ```

  `type` is `http_error` for explicit failures (404, 401, 409, 429…) and
  `validation_error` for a malformed body (with per-field `fields`).

## The contributor loop

| Step | Call | Auth |
|---|---|---|
| Share an explanation | `POST /documents` | ✅ |
| Submit evidence | `POST /evidence` | ✅ |
| Read a concept + provenance | `GET /concepts/{name}` | — |
| Browse concepts | `GET /concepts` | — |
| See open experiments | `GET /proposals` | — |
| Claim / resolve a proposal | `POST /proposals/{id}/claim` · `/resolve` | ✅ |
| Follow what changed | `GET /events` · `/events/stream` | — |
| Ask a grounded question | `GET /ask?q=…` | — |
| Trace any write | `GET /audit` | — |
| Register a webhook | `POST /webhooks` → `/approve` | ✅ |

The one rule that shapes the UX: **only evidence moves confidence.** A
document *proposes* a concept (capped below the belief threshold);
independent replication is what earns trust. Show confidence with its
breakdown (`contributors`, `independent_replications`, the `packages`
behind it) — never as a bare number.

## Python — the official client

Zero dependencies, ships with the engine:

```python
from allm.client import AllmClient, AllmError

allm = AllmClient("https://api.example", token="…")   # token only needed for writes

allm.submit_documents([{"name": "workshop-1", "text": "A plasma is an ionized gas…"}])
result = allm.submit_evidence(
    claim="a plasma lamp lights when energized",
    concept="plasma", contributor="ada", outcome="supported",   # supported | challenged | inconclusive
)
print(result["confidence"]["value"])

concept = allm.concept("plasma")          # {concept, provenance, evidential_confidence, …}
for event in allm.catch_up():             # drain the live feed
    print(event["type"], event["subject"])

try:
    allm.concept("nope")
except AllmError as e:
    print(e.status, e.type, e.message)    # 404 http_error "unknown concept 'nope'"
```

A full runnable walk-through is [`examples/81_client_end_to_end.py`](../examples/81_client_end_to_end.py).

## A browser client (JS)

Reads and the live feed are plain `fetch` / `EventSource`; writes add the
bearer header. Set `ALLM_API_CORS_ORIGINS` on the server to your origin.

```js
const BASE = "https://api.example";

// read a concept with provenance (open)
const concept = await (await fetch(`${BASE}/concepts/plasma`)).json();

// submit evidence (write — needs the token)
await fetch(`${BASE}/evidence`, {
  method: "POST",
  headers: { "Content-Type": "application/json", "Authorization": `Bearer ${TOKEN}` },
  body: JSON.stringify({ claim: "…", concept: "plasma", contributor: "ada", outcome: "supported" }),
});

// live updates — the browser auto-reconnects and resumes via Last-Event-ID,
// so you never miss or replay an event
const feed = new EventSource(`${BASE}/events/stream`);
feed.addEventListener("confidence.changed", (e) => applyConfidence(JSON.parse(e.data)));
feed.addEventListener("proposal.opened", (e) => showProposal(JSON.parse(e.data)));
```

Three ways to read the feed, same `seq` semantics: poll `GET /events?since=`,
subscribe to **SSE** `GET /events/stream`, or register a **webhook**
(server pushes to you; delivery is approval-gated and HMAC-signed — see
[`wire-format.md`](wire-format.md)). Event types today:
`evidence.submitted`, `confidence.changed`, `proposal.opened`,
`proposal.resolved`, `contribution.*`, `workshop.observed` — treat the set
as open and ignore any you don't handle.

## Ask ALLM — grounded answers

`GET /ask?q=…` (open) returns an answer built **only** from the evidence
graph — never a generative guess. It is **intent-aware**: a *how-to*
question is answered by the reproducible procedure behind the evidence,
not the definition.

```json
{ "found": true, "intent": "how_to", "status": "procedure", "concept": "The Nano Coating",
  "answer": "Here's how to make the nano coating — a procedure reproduced by 3 contributor(s), 2 independent replication(s) (confidence 0.79):",
  "steps": ["Degrease and lightly sand a metal plate", "Submerge it in a caustic (NaOH) bath", "Apply a low DC voltage", "Leave it 12 hours until a dark nano layer forms", "Rinse and dry"],
  "confidence": 0.79, "contributors": 3, "independent_replications": 2,
  "provenance": "…", "sources": ["pkg_…"] }
```

- `intent`: `how_to` · `quantity` · `definition`.
- `status`: `established` · `emerging` · `contested` · `unfounded` ·
  `procedure` (how-to with steps) · `no_procedure` (how-to, none
  submitted — it won't invent them) · `unknown` (nothing matched).
- `steps`: the reproducible procedure, present for answered how-to
  questions — render it as an ordered list.

When nothing matches, `found` is `false` and the answer says so — render
that honestly (it's the product's whole point), and use `status` to
colour the confidence. `AllmClient.ask("…")` wraps it; the reference chat
UI is served at `/chat`.

## What the client never sees

Identity, rewards and file storage are the platform's. The core takes
opaque `contributor` ids and artifact URIs; it never mints, ranks or pays
them. That boundary is what keeps replication-aware confidence honest —
popularity cannot move belief.
