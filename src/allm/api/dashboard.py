"""The system dashboard: one read-only view of the whole engine (M50/M51).

Everything the system knows about itself, gathered from the append-only
store without measuring or mutating anything:

- **KEL scorecard** — the eight epistemic-health metrics, their latest
  value and their trend over the recorded time series (sparklines).
- **Failure findings** — KEL's live diagnosis (KEL.md section 9). Green
  means no failure mode currently detected; that is the correctness
  signal.
- **Knowledge / evidence / proposals / contributions** — how much the
  ecosystem holds and in what state, including the human-approval ledger.
- **Population census** — per-namespace record counts. A namespace that
  is wired but empty is the honest "something is missing" signal, read
  straight from the data rather than a hand-kept checklist.

`system_state()` is a pure function of the store so it can be tested
without HTTP; the router just serves it as JSON plus a self-contained
HTML page (same posture as the Teacher visual UI).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

import allm
from allm.evidence import EvidenceBinder, EvidenceLedger
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.practice.contribution import ContributionBoard
from allm.proposals import ProposalBoard
from allm.storage.base import RecordStore
from allm.teacher import KnowledgeState

# label, one-line meaning, and whether higher is healthier (None = neutral).
KEL_METRICS: tuple[tuple[str, str, str, bool | None], ...] = (
    ("ghs", "Graph Health", "composite epistemic health score", True),
    ("rcr", "Redundancy Collapse", "duplicate concepts folded together", True),
    ("cd", "Conflict Density", "unresolved conflicts per concept", False),
    ("gst", "Graph Stability", "structure preserved vs last snapshot", True),
    ("crr", "Concept Reuse", "mean downstream uses per concept", True),
    ("lg", "Learning Gain", "mean confidence delta from teaching", True),
    ("cre", "Conflict Resolution", "conflicts that became learning", True),
    ("egr", "Evidence Growth", "better-founded knowledge over time", True),
    ("ks", "Knowledge Stability", "mastery held across later learning", True),
)

_HISTORY_POINTS = 40  # sparkline width; older points fall off the left


def system_state(store: RecordStore, *, audit_limit: int = 25) -> dict:
    """Gather the whole read-only picture from one record store."""
    graph = KnowledgeGraph(store)
    ledger = EvidenceLedger(store)
    binder = EvidenceBinder(graph, ledger)
    board = ProposalBoard(store, binder)
    contributions = ContributionBoard(store)
    state = KnowledgeState(store)
    kel = KnowledgeEvaluationLayer(graph, store, state, ledger=ledger)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": allm.__version__,
        "kel": _kel_view(kel),
        "graph": _graph_view(graph),
        "evidence": _evidence_view(ledger),
        "proposals": _proposals_view(board),
        "contributions": _contributions_view(contributions),
        "census": [
            {
                "namespace": s.namespace,
                "keys": s.keys,
                "records": s.records,
                "last_write": s.last_write.isoformat() if s.last_write else None,
            }
            for s in store.namespaces()
        ],
        "audit": [
            {
                "namespace": r.namespace,
                "key": r.key,
                "version": r.version,
                "reason": r.reason,
                "created_at": r.created_at.isoformat(),
            }
            for r in store.audit(limit=audit_limit)
        ],
    }


def _kel_view(kel: KnowledgeEvaluationLayer) -> dict:
    metrics = []
    for name, label, description, higher_is_better in KEL_METRICS:
        series = kel.history(name)
        values = [round(v, 4) for _, v in series][-_HISTORY_POINTS:]
        metrics.append(
            {
                "name": name,
                "label": label,
                "description": description,
                "higher_is_better": higher_is_better,
                "latest": values[-1] if values else None,
                "trend": kel.trend(name),
                "history": values,
                "measurements": len(series),
            }
        )
    findings = [f.model_dump() for f in kel.diagnose()]
    return {"metrics": metrics, "findings": findings}


def _graph_view(graph: KnowledgeGraph) -> dict:
    concepts = graph.concepts()
    active = [c for c in concepts if c.status == "active"]
    buckets = Counter()
    for c in active:
        edge = min(int(c.confidence * 4), 3)  # 0-.25,.25-.5,.5-.75,.75-1
        buckets[edge] += 1
    mean_conf = round(sum(c.confidence for c in active) / len(active), 4) if active else None
    top = sorted(active, key=lambda c: (c.usefulness, c.confidence), reverse=True)[:12]
    return {
        "total": len(concepts),
        "active": len(active),
        "retracted": len(concepts) - len(active),
        "mean_confidence": mean_conf,
        "confidence_buckets": [buckets.get(i, 0) for i in range(4)],
        "top_concepts": [
            {
                "name": c.name,
                "confidence": round(c.confidence, 3),
                "usefulness": round(c.usefulness, 3),
                "evidence_count": len(c.evidence),
            }
            for c in top
        ],
    }


def _evidence_view(ledger: EvidenceLedger) -> dict:
    packages = ledger.all_packages()
    return {
        "total": len(packages),
        "by_kind": dict(Counter(p.kind for p in packages)),
        "by_outcome": dict(Counter(p.outcome for p in packages)),
        "contributors": len({p.contributor for p in packages}),
        "concepts_covered": len({p.concept for p in packages}),
        "replications": sum(1 for p in packages if p.replicates),
    }


def _proposals_view(board: ProposalBoard) -> dict:
    proposals = board.proposals()
    return {
        "total": len(proposals),
        "by_status": dict(Counter(p.status for p in proposals)),
    }


def _contributions_view(contributions: ContributionBoard) -> dict:
    rows = contributions.all()
    return {
        "total": len(rows),
        "by_status": dict(Counter(c.status for c in rows)),
        "recent": [
            {
                "id": c.id,
                "status": c.status,
                "file": c.patch.file,
                "author": c.patch.author,
                "reviewer": c.reviewer,
                "trial_outcome": c.trial_outcome[:80],
            }
            for c in sorted(rows, key=lambda c: c.proposed_at, reverse=True)[:10]
        ],
    }


def snapshot_html(store: RecordStore) -> str:
    """A standalone, offline copy of the dashboard with today's state baked in.

    Freeze the system's self-view to a single file — no server, no
    network — so it can be archived and compared later. Same page as the
    live dashboard; the state is inlined instead of fetched.
    """
    import json

    state = json.dumps(system_state(store))
    inject = f"<script>window.__ALLM_STATE__ = {state};</script>\n"
    return DASHBOARD_HTML.replace("<script>", inject + "  <script>", 1)


def build_dashboard_router(store: RecordStore) -> APIRouter:
    """System dashboard routes bound to one record store (reads only)."""
    router = APIRouter(tags=["dashboard"])

    @router.get("/dashboard/state")
    def dashboard_state() -> dict:
        return system_state(store)

    @router.get("/dashboard", response_class=HTMLResponse)
    def dashboard_ui() -> str:
        return DASHBOARD_HTML

    return router


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ALLM — System Dashboard</title>
  <style>
    :root {
      --bg: #0b1020; --panel: #131a2e; --line: #263149; --text: #e6ebf5;
      --muted: #8a97b3; --good: #34d399; --warn: #fbbf24; --bad: #f87171;
      --accent: #60a5fa;
    }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; background: var(--bg);
      color: var(--text); }
    header { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;
      padding: 1rem 1.5rem; border-bottom: 1px solid var(--line); position: sticky;
      top: 0; background: var(--bg); z-index: 5; }
    header h1 { margin: 0; font-size: 1.15rem; }
    header .muted { color: var(--muted); font-size: 0.85rem; }
    header .spacer { flex: 1; }
    button { background: var(--panel); color: var(--text); border: 1px solid var(--line);
      border-radius: 6px; padding: 0.4rem 0.8rem; cursor: pointer; font-size: 0.85rem; }
    button:hover { border-color: var(--accent); }
    main { padding: 1.25rem 1.5rem; display: grid; gap: 1.25rem;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); max-width: 1500px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
      padding: 1rem 1.1rem; }
    .panel h2 { margin: 0 0 0.15rem; font-size: 0.95rem; }
    .panel .sub { color: var(--muted); font-size: 0.78rem; margin-bottom: 0.8rem; }
    .wide { grid-column: 1 / -1; }
    .metrics { display: grid; gap: 0.6rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem 0.7rem; }
    .metric .label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase;
      letter-spacing: 0.03em; }
    .metric .val { font-size: 1.5rem; font-weight: 600; margin: 0.1rem 0; }
    .metric .val.na { color: var(--muted); font-weight: 400; font-size: 1rem; }
    .metric .desc { font-size: 0.72rem; color: var(--muted); }
    .trend { font-size: 0.75rem; margin-left: 0.35rem; }
    .up { color: var(--good); } .down { color: var(--bad); } .flat { color: var(--muted); }
    .spark { display: block; margin-top: 0.35rem; }
    .findings .none { color: var(--good); }
    .finding { border-left: 3px solid var(--bad); padding: 0.4rem 0.7rem; margin: 0.4rem 0;
      background: rgba(248,113,113,0.08); border-radius: 0 6px 6px 0; }
    .finding .mode { font-weight: 600; color: var(--bad); font-size: 0.85rem; }
    .finding .detail { font-size: 0.82rem; color: var(--text); }
    .stat-row { display: flex; gap: 1.2rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
    .stat { }
    .stat .n { font-size: 1.35rem; font-weight: 600; }
    .stat .k { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; }
    .chips { display: flex; gap: 0.4rem; flex-wrap: wrap; }
    .chip { font-size: 0.75rem; background: var(--bg); border: 1px solid var(--line);
      border-radius: 20px; padding: 0.15rem 0.6rem; }
    .chip b { color: var(--accent); }
    table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    th, td { text-align: left; padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--line);
      white-space: nowrap; }
    th { color: var(--muted); font-weight: 500; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .scroll { overflow-x: auto; }
    .bars { display: flex; align-items: flex-end; gap: 4px; height: 46px; }
    .bar { flex: 1; background: var(--accent); border-radius: 3px 3px 0 0; min-height: 2px; }
    .bar-labels { display: flex; gap: 4px; margin-top: 3px; }
    .bar-labels span { flex: 1; text-align: center; font-size: 0.65rem; color: var(--muted); }
    .empty { color: var(--warn); }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
    a { color: var(--accent); }
  </style>
</head>
<body>
  <header>
    <h1>ALLM · System Dashboard</h1>
    <span class="muted" id="meta">loading…</span>
    <span class="spacer"></span>
    <label class="muted"><input type="checkbox" id="auto" /> auto-refresh (15s)</label>
    <button id="refresh">Refresh</button>
  </header>
  <main id="root"></main>
  <script>
    const $ = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c;
      if (h != null) e.innerHTML = h; return e; };
    const fmt = v => v == null ? "—" : (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v);

    function sparkline(values, higher) {
      const w = 130, h = 30, n = values.length;
      if (n < 2) return `<svg class="spark" width="${w}" height="${h}"></svg>`;
      const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
      const pts = values.map((v, i) =>
        `${(i / (n - 1) * (w - 2) + 1).toFixed(1)},${(h - 1 - (v - lo) / span * (h - 2)).toFixed(1)}`
      ).join(" ");
      const last = values[n - 1], first = values[0];
      const col = higher == null ? "var(--accent)"
        : (last >= first) === higher ? "var(--good)" : "var(--bad)";
      return `<svg class="spark" width="${w}" height="${h}">
        <polyline fill="none" stroke="${col}" stroke-width="1.5" points="${pts}"/></svg>`;
    }

    function metricTile(m) {
      let trend = "";
      if (m.trend != null && m.trend !== 0) {
        const good = m.higher_is_better == null ? "flat"
          : (m.trend > 0) === m.higher_is_better ? "up" : "down";
        trend = `<span class="trend ${good}">${m.trend > 0 ? "▲" : "▼"} ${Math.abs(m.trend).toFixed(3)}</span>`;
      }
      const val = m.latest == null
        ? `<div class="val na">not measured</div>`
        : `<div class="val">${fmt(m.latest)}${trend}</div>`;
      return `<div class="metric">
        <div class="label">${m.label}</div>
        ${val}
        ${sparkline(m.history, m.higher_is_better)}
        <div class="desc">${m.description}</div>
      </div>`;
    }

    function chips(obj) {
      const keys = Object.keys(obj);
      if (!keys.length) return `<span class="muted">none yet</span>`;
      return `<div class="chips">${keys.map(k => `<span class="chip">${k} <b>${obj[k]}</b></span>`).join("")}</div>`;
    }

    function panel(title, sub, bodyHtml, wide) {
      return `<section class="panel ${wide ? "wide" : ""}"><h2>${title}</h2>
        <div class="sub">${sub}</div>${bodyHtml}</section>`;
    }

    function render(s) {
      document.getElementById("meta").textContent =
        `v${s.version} · ${new Date(s.generated_at).toLocaleString()}`;
      const root = document.getElementById("root");
      root.innerHTML = "";
      const html = [];

      // KEL scorecard
      html.push(panel("KEL Scorecard",
        "epistemic health, latest value and trend over every recorded measurement",
        `<div class="metrics">${s.kel.metrics.map(metricTile).join("")}</div>`, true));

      // Findings
      const f = s.kel.findings;
      const findingsBody = f.length
        ? f.map(x => `<div class="finding"><div class="mode">${x.mode.replace(/_/g, " ")}</div>
            <div class="detail">${x.detail}</div></div>`).join("")
        : `<p class="none"><span class="dot" style="background:var(--good)"></span>No failure modes detected in the latest measurements.</p>`;
      html.push(panel("Failure Findings", "KEL live diagnosis (KEL.md §9) — this is the correctness signal",
        `<div class="findings">${findingsBody}</div>`));

      // Knowledge graph
      const g = s.graph, maxB = Math.max(1, ...g.confidence_buckets);
      const bars = g.confidence_buckets.map(v =>
        `<div class="bar" style="height:${v / maxB * 100}%" title="${v}"></div>`).join("");
      const topRows = g.top_concepts.map(c =>
        `<tr><td>${c.name}</td><td class="num">${c.confidence}</td><td class="num">${c.usefulness}</td><td class="num">${c.evidence_count}</td></tr>`).join("");
      html.push(panel("Knowledge Graph", "concepts, confidence spread and the planner's top picks",
        `<div class="stat-row">
           <div class="stat"><div class="n">${g.total}</div><div class="k">concepts</div></div>
           <div class="stat"><div class="n">${g.active}</div><div class="k">active</div></div>
           <div class="stat"><div class="n">${g.retracted}</div><div class="k">retracted</div></div>
           <div class="stat"><div class="n">${fmt(g.mean_confidence)}</div><div class="k">mean conf.</div></div>
         </div>
         <div class="bars">${bars}</div>
         <div class="bar-labels"><span>0–.25</span><span>.25–.5</span><span>.5–.75</span><span>.75–1</span></div>
         <div class="scroll" style="margin-top:0.7rem"><table>
           <tr><th>concept</th><th class="num">conf</th><th class="num">useful</th><th class="num">evid</th></tr>
           ${topRows || `<tr><td colspan="4" class="muted">no concepts yet</td></tr>`}
         </table></div>`));

      // Evidence
      const e = s.evidence;
      html.push(panel("Evidence Ledger", "ground truth backing the graph — documents propose, evidence disposes",
        `<div class="stat-row">
           <div class="stat"><div class="n">${e.total}</div><div class="k">packages</div></div>
           <div class="stat"><div class="n">${e.contributors}</div><div class="k">contributors</div></div>
           <div class="stat"><div class="n">${e.replications}</div><div class="k">replications</div></div>
           <div class="stat"><div class="n">${e.concepts_covered}</div><div class="k">concepts covered</div></div>
         </div>
         <div class="sub">by kind</div>${chips(e.by_kind)}
         <div class="sub" style="margin-top:0.5rem">by outcome</div>${chips(e.by_outcome)}`));

      // Proposals + contributions
      const p = s.proposals, c = s.contributions;
      const contribRows = c.recent.map(r =>
        `<tr><td>${r.status}</td><td>${r.file}</td><td>${r.author}</td><td>${r.reviewer || "—"}</td></tr>`).join("");
      html.push(panel("Proposals & Contributions",
        "open experiments and the human-approval ledger (nothing applies without a named reviewer)",
        `<div class="stat-row">
           <div class="stat"><div class="n">${p.total}</div><div class="k">proposals</div></div>
           <div class="stat"><div class="n">${c.total}</div><div class="k">contributions</div></div>
         </div>
         <div class="sub">proposals by status</div>${chips(p.by_status)}
         <div class="sub" style="margin-top:0.5rem">contributions by status</div>${chips(c.by_status)}
         <div class="scroll" style="margin-top:0.7rem"><table>
           <tr><th>status</th><th>file</th><th>author</th><th>reviewer</th></tr>
           ${contribRows || `<tr><td colspan="4" class="muted">no contributions yet</td></tr>`}
         </table></div>`));

      // Census
      const censusRows = s.census.map(r => {
        const empty = r.keys === 0 ? ' class="empty"' : "";
        return `<tr><td${empty}>${r.namespace}</td><td class="num">${r.keys}</td><td class="num">${r.records}</td><td class="muted">${r.last_write ? new Date(r.last_write).toLocaleString() : "—"}</td></tr>`;
      }).join("");
      html.push(panel("Population Census", "every storage namespace and how much it holds — empty = wired but never exercised",
        `<div class="scroll"><table>
           <tr><th>namespace</th><th class="num">keys</th><th class="num">records</th><th>last write</th></tr>
           ${censusRows || `<tr><td colspan="4" class="muted">store is empty</td></tr>`}
         </table></div>`));

      // Audit
      const auditRows = s.audit.map(r =>
        `<tr><td class="muted">${new Date(r.created_at).toLocaleTimeString()}</td><td>${r.namespace}/${r.key}</td><td class="num">v${r.version}</td><td>${r.reason || ""}</td></tr>`).join("");
      html.push(panel("Recent Writes", "the append-only trail, newest first — nothing overwritten, everything attributable",
        `<div class="scroll"><table>
           <tr><th>time</th><th>record</th><th class="num">ver</th><th>reason</th></tr>
           ${auditRows || `<tr><td colspan="4" class="muted">no writes yet</td></tr>`}
         </table></div>`, true));

      root.innerHTML = html.join("");
    }

    async function load() {
      if (window.__ALLM_STATE__) { render(window.__ALLM_STATE__); return; }  // snapshot mode
      document.getElementById("meta").textContent = "loading…";
      try {
        const res = await fetch("/dashboard/state");
        render(await res.json());
      } catch (err) {
        document.getElementById("meta").textContent = "failed to load: " + err;
      }
    }

    let timer = null;
    document.getElementById("refresh").onclick = load;
    document.getElementById("auto").onchange = (ev) => {
      clearInterval(timer);
      if (ev.target.checked) timer = setInterval(load, 15000);
    };
    load();
  </script>
</body>
</html>
"""
