# Operator quickstart

Zero to a live, seeded ALLM you can talk to — in about five minutes. Each
step links to the deeper doc when you need more.

## 1. Run it

```bash
export ALLM_API_TOKEN=$(openssl rand -hex 24)   # your write token — keep it
docker compose up --build                        # auth on, data on a volume
```

Compose refuses to start without a token, so the API is never
accidentally open. → [deploy.md](deploy.md)

## 2. Check it's up

```bash
curl -s localhost:8000/health   # {"status":"ok",...}     liveness
curl -s localhost:8000/ready    # {"status":"ready",...}  store answers
```

## 3. Seed the demo (optional but recommended)

Run the whole public loop once so the dashboard isn't empty:

```bash
docker compose exec api allm seed --db /data/allm.sqlite3
```

Now open **`localhost:8000/dashboard`** — a resolved contested claim
(`The Nano Coating`, confidence 0.41 → 0.79), evidence with replications,
KEL metrics, and a live feed. That is the exit-criterion loop, already
run once.

Or open **`localhost:8000/chat`** and ask *"how long does the nano
coating take to form?"* — a grounded answer with its confidence and
provenance, and an honest *"no evidence yet"* for anything the base can't
back.

## 4. Talk to it

Reads are open; writes need the token.

```bash
# submit evidence (write — bearer token). Creates the concept on first mention.
curl -s -X POST localhost:8000/evidence \
  -H "Authorization: Bearer $ALLM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"claim":"a plasma lamp lights when energized","concept":"plasma","contributor":"you","outcome":"supported"}'

# read it back with its provenance (open)
curl -s localhost:8000/concepts/plasma | python -m json.tool

# watch the live feed (Ctrl-C to stop) — your write is already on it
curl -N localhost:8000/events/stream
```

## 5. Point a client at it

Python (zero-dependency, ships with the engine):

```python
from allm.client import AllmClient
allm = AllmClient("http://localhost:8000", token="…")
allm.submit_evidence(claim="…", concept="plasma", contributor="me", outcome="supported")
```

Browser: set `ALLM_API_CORS_ORIGINS` to your frontend's origin, then
`fetch` for reads/writes and `new EventSource("/events/stream")` for live
updates. → [client-guide.md](client-guide.md) ·
contract: [wire-format.md](wire-format.md)

## Before real users

- **Real token, TLS in front.** Terminate HTTPS at a reverse proxy; set a
  strong `ALLM_API_TOKEN`.
- **CORS is off by default** — set `ALLM_API_CORS_ORIGINS` to exact
  origins, not `*`.
- **Turn on backups:** `docker compose --profile backup up -d`.
- The full trust model and operator setup: [security-review.md](security-review.md).

## The one rule to build around

**Only evidence moves confidence.** A document *proposes* a claim
(capped below the belief threshold); independent replication is what
earns trust. Show confidence with its breakdown — contributors,
replications, the packages behind it — never as a bare number. Popularity
cannot move belief, by design.
