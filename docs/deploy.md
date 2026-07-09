# Deploying ALLM (M52 — deployable core)

From a clone to a running, authenticated API whose data survives a
restart — in one command. The container serves the HTTP boundary only;
the heavy ML extras are never installed (everything imports them lazily),
so the image is small and needs no GPU.

## One command

```bash
export ALLM_API_TOKEN=$(openssl rand -hex 24)   # keep this; it is your auth
docker compose up --build
```

Compose **refuses to start without `ALLM_API_TOKEN`** — auth is on by
default, so the API is never accidentally exposed open. The store lives
on the `allm-data` volume and survives `docker compose down` (use
`down -v` to wipe it).

Verify it's up:

```bash
curl -s localhost:8000/health                     # liveness  -> {"status":"ok",...}
curl -s localhost:8000/ready                       # readiness -> {"status":"ready",...}
curl -s localhost:8000/wire | head                 # the frozen wire contract (open)
curl -s localhost:8000/dashboard                    # the system dashboard (open read)

# writes need the token
curl -s -X POST localhost:8000/evidence \
  -H "Authorization: Bearer $ALLM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"claim":"plasma is hot","concept":"plasma","contributor":"you","outcome":"supported"}'
```

`/health` is **liveness** (the process is up, never touches the store);
`/ready` is **readiness** (the store answers, safe to route traffic).
Compose gates the container's health on `/ready`.

### Optional: seed a demo

A fresh store is empty. To make the dashboard immediately alive — and to
watch the whole public loop run once (discussion → conflict → proposal →
replication → confidence shift) — seed it:

```bash
docker compose exec api allm seed --db /data/allm.sqlite3
```

Then open `/dashboard`: concepts, a resolved contested claim, evidence
with replications, KEL metrics and a populated live feed. Re-seeding an
existing store needs `--force`.

## Configuration (all via `ALLM_` env)

| Variable | Default | Meaning |
|---|---|---|
| `ALLM_API_TOKEN` | *(required)* | Bearer token for every write. Rotate by changing it and restarting. |
| `ALLM_STORAGE__PATH` | `/data/allm.sqlite3` | Store location (on the volume). |
| `ALLM_API_RATE_LIMIT` | `60/60` | Per-principal token bucket (`requests/seconds`). |
| `ALLM_API_CORS_ORIGINS` | *(none)* | Comma-separated browser origins allowed cross-site (e.g. `https://app.example`). Empty = same-origin only. `*` = any (dev). |
| `ALLM_PORT` | `8000` | Host port to publish. |
| `ALLM_LOG_LEVEL` | `INFO` | Log level. |

### A browser client

Set `ALLM_API_CORS_ORIGINS` to your frontend's origin(s) so it can call
the API cross-site. For live updates, subscribe to the **Server-Sent
Events** feed instead of polling:

```js
const es = new EventSource("https://api.example/events/stream");
es.addEventListener("confidence.changed", (e) => update(JSON.parse(e.data)));
// the browser auto-reconnects and resends Last-Event-ID, so no event is missed
```

Reads and the SSE stream are open; only writes need the bearer token.

Reads (`/concepts`, `/events`, `/dashboard`, `/wire`, `/audit`) are open
by design; every write requires the bearer token and is rate-limited.

## Data: persistence, backup, restore

Data is on the `allm-data` Docker volume. Turn on scheduled online
backups (consistent even while the API keeps writing) with the opt-in
profile:

```bash
docker compose --profile backup up -d     # writes /backups every ALLM_BACKUP_INTERVAL (default 3600s)
```

Manual backup / verify / restore, any time:

```bash
docker compose exec api allm db backup  --db /data/allm.sqlite3 /data/manual-backup.sqlite3
docker compose exec api allm db verify  --db /data/manual-backup.sqlite3
docker compose exec api allm db restore --db /data/allm.sqlite3 /data/manual-backup.sqlite3 --force
```

Restore never clobbers silently — it verifies the backup first and keeps
the displaced file as `.replaced`.

## Without Docker

```bash
pip install -e ".[api]"
export ALLM_API_TOKEN=$(openssl rand -hex 24)
export ALLM_STORAGE__PATH=./data/allm.sqlite3
uvicorn --factory allm.api.app:create_default_app --host 0.0.0.0 --port 8000
```

## Before you put it on the internet

- **TLS terminates in front.** Run behind a reverse proxy (Caddy, nginx,
  a cloud LB) that does HTTPS; the container speaks plain HTTP inside the
  network.
- **Rate limiting is per-instance** (in-process). Behind more than one
  replica, add a shared limiter or sticky routing — see
  [`security-review.md`](security-review.md).
- **CORS is off by default** — set `ALLM_API_CORS_ORIGINS` to your
  frontend's exact origin(s) rather than `*` in production.
- The full trust model — what each endpoint defends and the operator
  setup for strangers — is in [`security-review.md`](security-review.md).
