"""Ask ALLM — a minimal grounded chat UI (Roadmap M52).

Served like the dashboard: a self-contained page that calls ``/ask`` and
renders each answer with its status, confidence and provenance. The point
of the interface *is* the grounding — an answer you can trace, or an
honest "no evidence yet", never a fluent guess. A reference the frontend
team can copy.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


def build_chat_router() -> APIRouter:
    """Serve the static Ask-ALLM page (the data comes from ``/ask``)."""
    router = APIRouter(tags=["chat"])

    @router.get("/chat", response_class=HTMLResponse)
    def chat_ui() -> str:
        return CHAT_HTML

    return router


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ask ALLM</title>
  <style>
    :root {
      --bg: #0b1020; --panel: #131a2e; --line: #263149; --text: #e6ebf5;
      --muted: #8a97b3; --good: #34d399; --warn: #fbbf24; --bad: #f87171;
      --accent: #60a5fa; --user: #1e293b;
    }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; background: var(--bg);
      color: var(--text); display: flex; flex-direction: column; height: 100vh; }
    header { padding: 0.9rem 1.25rem; border-bottom: 1px solid var(--line); }
    header h1 { margin: 0; font-size: 1.05rem; }
    header p { margin: 0.2rem 0 0; color: var(--muted); font-size: 0.8rem; }
    #log { flex: 1; overflow-y: auto; padding: 1.25rem; display: flex;
      flex-direction: column; gap: 0.9rem; max-width: 820px; width: 100%;
      margin: 0 auto; }
    .msg { max-width: 88%; }
    .msg.user { align-self: flex-end; background: var(--user); border: 1px solid var(--line);
      padding: 0.5rem 0.8rem; border-radius: 12px 12px 2px 12px; }
    .msg.allm { align-self: flex-start; background: var(--panel); border: 1px solid var(--line);
      padding: 0.7rem 0.9rem; border-radius: 12px 12px 12px 2px; }
    .answer { line-height: 1.5; }
    .badges { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.55rem; align-items: center; }
    .badge { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em;
      padding: 0.12rem 0.5rem; border-radius: 20px; border: 1px solid var(--line); }
    .badge.established { color: var(--good); border-color: var(--good); }
    .badge.emerging { color: var(--accent); border-color: var(--accent); }
    .badge.contested { color: var(--warn); border-color: var(--warn); }
    .badge.unfounded { color: var(--bad); border-color: var(--bad); }
    .badge.unknown { color: var(--muted); }
    .meta { font-size: 0.75rem; color: var(--muted); }
    details { margin-top: 0.5rem; }
    summary { cursor: pointer; font-size: 0.75rem; color: var(--accent); }
    pre { white-space: pre-wrap; font-size: 0.75rem; color: var(--muted);
      background: var(--bg); border: 1px solid var(--line); border-radius: 6px;
      padding: 0.5rem; margin: 0.4rem 0 0; overflow-x: auto; }
    .suggestion { margin-top: 0.5rem; font-size: 0.82rem; color: var(--accent); }
    .related { margin-top: 0.4rem; font-size: 0.75rem; color: var(--muted); }
    form { display: flex; gap: 0.5rem; padding: 0.9rem 1.25rem; border-top: 1px solid var(--line);
      max-width: 820px; width: 100%; margin: 0 auto; }
    input { flex: 1; background: var(--panel); border: 1px solid var(--line);
      color: var(--text); border-radius: 8px; padding: 0.6rem 0.8rem; font-size: 0.95rem; }
    input:focus { outline: 2px solid var(--accent); border-color: var(--accent); }
    button { background: var(--accent); color: #06122b; border: none; border-radius: 8px;
      padding: 0 1.1rem; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: default; }
  </style>
</head>
<body>
  <header>
    <h1>Ask ALLM</h1>
    <p>Answers come only from submitted evidence — with their confidence and provenance. No evidence, no claim.</p>
  </header>
  <div id="log"></div>
  <form id="form">
    <input id="q" autocomplete="off" placeholder="Ask about a concept… e.g. how long does the nano coating take?" />
    <button id="send" type="submit">Ask</button>
  </form>
  <script>
    // Works at /chat or under a proxy prefix like /allm/chat.
    const PREFIX = location.pathname.replace(/\\/chat\\/?$/, "");
    const log = document.getElementById("log");
    const input = document.getElementById("q");
    const send = document.getElementById("send");

    function el(cls, html) { const d = document.createElement("div"); d.className = cls;
      if (html != null) d.innerHTML = html; return d; }
    function esc(s) { return (s || "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

    function addUser(text) {
      log.appendChild(el("msg user", esc(text)));
      log.scrollTop = log.scrollHeight;
    }

    function addAnswer(a) {
      const parts = [`<div class="answer">${esc(a.answer)}</div>`];
      const badges = [`<span class="badge ${a.status}">${a.status}</span>`];
      if (a.confidence != null) badges.push(
        `<span class="meta">confidence ${a.confidence.toFixed(2)} · ` +
        `${a.contributors} contributor(s) · ${a.independent_replications} replication(s)</span>`);
      parts.push(`<div class="badges">${badges.join("")}</div>`);
      if (a.open_questions && a.open_questions.length)
        parts.push(`<div class="related">open question: ${esc(a.open_questions[0])}</div>`);
      if (a.related && a.related.length)
        parts.push(`<div class="related">related: ${a.related.map(esc).join(", ")}</div>`);
      if (a.suggestion) parts.push(`<div class="suggestion">${esc(a.suggestion)}</div>`);
      if (a.provenance) parts.push(
        `<details><summary>provenance${a.sources && a.sources.length ? ` · ${a.sources.length} package(s)` : ""}</summary>` +
        `<pre>${esc(a.provenance)}</pre></details>`);
      log.appendChild(el("msg allm", parts.join("")));
      log.scrollTop = log.scrollHeight;
    }

    // Offline demo mode: if answers are inlined (window.__ALLM_DEMO__), match
    // locally instead of calling /ask — used for a self-contained snapshot.
    function demoAnswer(q) {
      const qt = new Set((q.toLowerCase().match(/[a-z0-9]+/g) || []));
      let best = null, score = 0;
      for (const item of window.__ALLM_DEMO__) {
        const s = item.tokens.filter(t => qt.has(t)).length;
        if (s > score) { score = s; best = item; }
      }
      return score > 0 ? best.answer : {
        status: "unknown", found: false, confidence: null,
        answer: "I don't have evidence about that yet — nothing matches your question. I won't guess.",
        suggestion: "In the live system you'd contribute a document or evidence package.",
      };
    }

    async function respond(q) {
      if (window.__ALLM_DEMO__) return demoAnswer(q);
      const res = await fetch(PREFIX + "/ask?q=" + encodeURIComponent(q));
      return res.json();
    }

    async function ask(q) {
      addUser(q);
      input.value = ""; send.disabled = true;
      try {
        addAnswer(await respond(q));
      } catch (err) {
        log.appendChild(el("msg allm", `<div class="answer">Couldn't reach the knowledge base: ${esc(String(err))}</div>`));
      } finally {
        send.disabled = false; input.focus();
      }
    }

    document.getElementById("form").onsubmit = (ev) => {
      ev.preventDefault();
      const q = input.value.trim();
      if (q) ask(q);
    };
    window.allmAsk = ask;  // demo pages can drive an opening exchange

    input.focus();
  </script>
</body>
</html>
"""
