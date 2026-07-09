# Security review — the three dangerous paths (M50)

*Reviewed 2026-07-09, against the code at v0.9.0. This is the second M50
exit criterion: a written assessment of every path where untrusted input
reaches an execution or a trust boundary.*

ALLM is a **research platform, not a hardened multi-tenant service.** The
goal of this review is not to claim the system is safe to expose raw to
the open internet — it is to state precisely what each dangerous path
defends against, what it does not, and what an operator must add before
taking it public. Honesty about the boundary is the security property.

The three paths, in order of exposure:

1. **Evidence submission** — untrusted *data* over HTTP.
2. **Code grading** — untrusted *code*, run to score it.
3. **Practice execution** — untrusted *code*, run to learn from it.

---

## 1. Evidence submission (untrusted data over HTTP)

**The path.** `POST /evidence`, `/proposals`, `/relations` — a stranger
submits a claim, an outcome, a contributor id. It becomes an
append-only record and can shift a confidence value.

**What defends it.**

- **No overwrite, ever.** The storage layer is append-only and versioned
  with a `reason` per write (`storage/sqlite.py`). A malicious submission
  cannot destroy or silently rewrite prior state; the worst it does is
  add a row, and every row is attributable and reversible.
- **Popularity cannot move belief.** Confidence is replication-aware:
  injected-document confidence is capped at 0.6 — below KEL's 0.75
  threshold — so *documents propose, evidence disposes*
  (`researcher/graph_injection.py`, the `unearned_confidence` detector).
  A thousand identical unfounded claims do not manufacture confidence;
  only independent replication does. This is the anti-gaming primitive,
  and it is a **belief-integrity** control, not just an access control.
- **Auth is a hook, identity is the platform's.** `TokenVerifier`
  (`api/security.py`) is the seam: the core verifies what it is handed
  and never invents an identity model. Two honest defaults —
  `AllowAllVerifier` (dev, loudly logged on every startup) and
  `StaticTokenVerifier` (`ALLM_API_TOKEN`, constant-time compare, ≥16
  chars enforced). Reads are open; **every write is guarded** (401).
- **Resource-exhaustion controls.** Per-principal token-bucket rate
  limiting (`ALLM_API_RATE_LIMIT`, 429 on exceed); input size caps on
  every write schema (413/422); pagination on every list endpoint so a
  read cannot ask for unbounded work.
- **Audit by construction.** The append-only store *is* the audit log;
  `GET /audit` exposes metadata only (never values), paginated.

**What it does NOT defend against, and the operator's job.**

- **`AllowAllVerifier` is not authentication.** Ship
  `StaticTokenVerifier` or a platform-provided verifier before exposure.
  The startup log says so on every boot; heed it.
- **Rate limiting is in-process.** The token bucket lives in the app
  process, so it is per-instance, not per-cluster. Behind a load
  balancer, put a shared limiter (or sticky routing) in front — the
  in-process limiter is a floor against a single noisy client, not a
  distributed quota.
- **Sybil identity is out of scope by design.** The core sees opaque
  contributor ids; making an id *cost* something (stake, reputation)
  is the platform's responsibility (M51). Replication-aware confidence
  is what keeps Sybils from moving belief even when they can move volume.

---

## 2. Code grading (untrusted code, run to score it)

**The path.** `CodingGrader` (`exam/coding.py`) takes a student's answer
— an arbitrary Python program — and executes it to compare its stdout
against the expected output. The "student" may be an untrusted model or,
in M49, an outside contributor.

**What defends it.**

- **No shell, no interpolation.** The program is passed as `argv` to
  `python -I` (isolated mode — no user site, no `PYTHON*` env influence,
  no implicit `sys.path[0]`). There is no shell to inject into.
- **Kernel rlimits** (`practice/limits.py`, `preexec_fn`): CPU seconds,
  address space, file size, and `RLIMIT_CORE=0`, applied in the child
  before `exec`. The kernel stops the CPU spin and the memory bomb that a
  wall-clock timeout alone cannot.
- **Wall-clock timeout** on top, as the backstop for anything that blocks
  without consuming CPU (e.g. sleeping).
- **No secret inheritance** *(fixed in this review — see below).* The
  child gets `clean_env()`, a minimal allowlist (`PATH`, `LANG`, `LC_ALL`,
  `TZ`, `HOME`, `TMPDIR`) — never the parent's `ALLM_API_TOKEN`,
  `OLLAMA_API_KEY`, or any other inherited secret.

**What it does NOT defend against.**

- **Same-uid execution.** The grader child runs as the same OS user as
  the parent. rlimits are the floor, not a jail: submitted code can read
  any file that user can read (minus the env secrets we now strip) and,
  crucially, **can reach the network** — `CodingGrader` does *not* apply
  bubblewrap. For grading genuinely untrusted third-party code, run the
  grader itself inside the practice engine's namespace isolation, or
  under a dedicated unprivileged uid in a container. This is stated in
  the module docstring as the deliberate research-platform posture.

---

## 3. Practice execution (untrusted code, run to learn from it)

**The path.** `SandboxExecutor.run()` (`practice/executor.py`) executes a
`PracticeProcedure` — the core of M48. Variables sweep across values;
each run's stdout becomes content-addressed ground truth. M49 points this
at *repository test suites* (`practice/repo_tasks.py`): running a repo's
tests executes that repo's code.

This is the most defended path, because it is the one designed from the
start to run code we did not write.

**What defends it (three layers, outermost first).**

1. **Namespace isolation (bubblewrap)** when available
   (`practice/isolation.py`): read-only root, private `/tmp`,
   `--unshare-all` (**no network**), `--die-with-parent`, and only the
   declared workdir bind-mounted writable. Kernel-enforced and **tested
   against a live local daemon**, not just mocked. Auto-detected with
   loud degradation: `auto` falls back to no-isolation with a warning;
   `bwrap` *requires* it and raises if unavailable, so an operator can
   demand isolation and fail closed.
2. **Kernel rlimits** — same as grading, roomier budget for repo trials
   (`REPO_TASK_LIMITS`: 300s / 2 GB / 64 MB).
3. **Timeout + `python -I` + literal variable binding** — values are
   `repr()`'d into an assignment prelude, never interpolated into code
   text (`bind_variables`), and non-literals are rejected. Crash and
   timeout are recorded *outcomes*, not exceptions.

- **Repo patches are tried in a disposable copy.** `trial_patch` copies
  the repo to a tempdir (ignoring `.git`, `.venv`, …), applies the
  candidate there, and rejects any absolute or `..`-escaping path. The
  working tree is never touched by a trial.
- **The M49 approval invariant.** Nothing leaves the system without a
  named-human approval record: `ContributionBoard.apply()` raises without
  one, and there is no push/remote/network code in the contribution path
  at all (structurally tested). Isolation stops runaway code; the
  approval gate stops *unattended outbound action*.
- **No secret inheritance** *(fixed in this review — see below)*, in both
  subprocess and bubblewrap modes.

**What it does NOT defend against.**

- **Same-uid inside the namespace.** Even with bubblewrap, code runs as
  the invoking user's uid mapped inside the namespace. The namespace
  removes the network and the writable filesystem; it does not add a
  second privilege boundary. For a hostile-by-assumption deployment, add
  a dedicated unprivileged uid and/or seccomp on top.
- **Isolation is best-effort under `auto`.** If bubblewrap is missing and
  the mode is `auto`, execution proceeds with a warning and rlimits+
  timeout only. Operators who require the namespace must set
  `ALLM_SANDBOX_ISOLATION=bwrap` so the system fails closed instead of
  degrading silently.

---

## Finding fixed during this review: environment-secret leakage

While probing the three paths, this review found a **real
vulnerability** and fixed it in the same pass.

**The leak.** Both `SandboxExecutor` and `CodingGrader` spawned children
*without* setting `env`, so the child inherited the parent's entire
environment. `python -I` isolates imports and `PYTHON*` variables but
**not** the environment dict, and bubblewrap passes the environment
through unchanged. A probe confirmed that
`os.environ.get("ALLM_API_TOKEN")` inside executed code returned the
parent's real token — in `none` *and* `bwrap` modes, and in the grader.
Any submitted program could read every secret the process held.

**The fix.** A shared `clean_env()` (`practice/limits.py`) returns a
minimal allowlist — `PATH`, `LANG`, `LC_ALL`, `TZ`, `HOME`, `TMPDIR` —
and both execution paths now pass it as `env=`. The probe now returns
`None` for every secret while `PATH` still resolves so programs run at
all. Locked in by a regression test
(`test_executed_code_cannot_read_parent_secrets`): parent sets
`ALLM_API_TOKEN`/`OLLAMA_API_KEY`, the sandboxed probe must report
`None None True`.

**Lesson for the invariant list.** Anything a submission genuinely needs
must be *supplied deliberately*, never inherited by accident. The
allowlist is the enforcement of that rule.

---

## Verdict

The exit criterion — "security review of the three dangerous paths" — is
met, and it uncovered and closed a live secret-leakage bug rather than
merely blessing the status quo.

**Safe to invite strangers to, with the stated operator setup:** real
verifier (not `AllowAllVerifier`), `ALLM_SANDBOX_ISOLATION=bwrap`,
shared rate limiting behind any load balancer, and — for grading
genuinely hostile code — a dedicated unprivileged uid or container around
the grader. The single unremoved caveat that no in-process control can
fix is **same-uid execution**; it is a deployment-topology decision, and
it is documented at every path rather than hidden.

The security property this system actually offers is not "impossible to
misuse" but "**nothing silent**": no overwrite, no unearned confidence,
no unattended outbound action, no degraded isolation without a warning,
and now no inherited secret. Every boundary announces itself.
