"""Namespace isolation via bubblewrap (Roadmap M50), offline."""

from pathlib import Path

import pytest

from allm.practice import (
    CandidatePatch,
    PracticeProcedure,
    SandboxExecutor,
    VariableSpec,
    bwrap_available,
    trial_patch,
)

needs_bwrap = pytest.mark.skipif(
    not bwrap_available(), reason="bubblewrap not functional here"
)


def probe(program: str, **kwargs) -> PracticeProcedure:
    return PracticeProcedure(
        id="iso-probe",
        description="isolation probe",
        program=program,
        variables=(VariableSpec(name="n", default=1),),
        **kwargs,
    )


@needs_bwrap
def test_network_is_kernel_blocked_inside_procedures() -> None:
    run = SandboxExecutor(isolation="bwrap").run(
        probe(
            "import urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2)\n"
            "    print('NETWORK REACHABLE')\n"
            "except Exception:\n"
            "    print('network blocked')\n"
        )
    )
    assert run.outcome == "network blocked"


@needs_bwrap
def test_filesystem_is_read_only_outside_the_workdir(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    run = SandboxExecutor(isolation="bwrap").run(
        probe(
            f"try:\n"
            f"    open({str(victim)!r}, 'w').write('clobbered')\n"
            f"    print('WRITE ALLOWED')\n"
            f"except Exception:\n"
            f"    print('write blocked')\n"
        )
    )
    assert run.outcome == "write blocked"
    assert victim.read_text() == "precious"


@needs_bwrap
def test_declared_workdir_stays_writable(tmp_path: Path) -> None:
    run = SandboxExecutor(isolation="bwrap").run(
        probe(
            "open('inside.txt', 'w').write('ok')\n"
            "print(open('inside.txt').read())\n",
            workdir=str(tmp_path),
        )
    )
    assert run.outcome == "ok"
    assert (tmp_path / "inside.txt").read_text() == "ok"


@needs_bwrap
def test_repo_trial_runs_fully_isolated(tmp_path: Path) -> None:
    repo = tmp_path / "calcrepo"
    repo.mkdir()
    (repo / "calculator.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_calculator.py").write_text(
        "from calculator import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    patch = CandidatePatch.build(
        file="calculator.py",
        content="def add(a, b):\n    return a + b\n",
        reasoning="fix",
        author="apprentice",
    )
    run = trial_patch(
        repo, patch, "test_calculator.py", executor=SandboxExecutor(isolation="bwrap")
    )
    assert run.outcome == "pass"


def test_auto_mode_always_resolves() -> None:
    executor = SandboxExecutor(isolation="auto")
    assert executor.isolation in ("bwrap", "none")
    run = executor.run(probe("print('hello')\n"))
    assert run.outcome == "hello"


def test_impossible_mode_fails_loudly(monkeypatch) -> None:
    import allm.practice.isolation as iso

    monkeypatch.setattr(iso, "bwrap_available", lambda: False)
    with pytest.raises(RuntimeError, match="not functional"):
        iso.resolve_isolation("bwrap")
    with pytest.raises(ValueError, match="unknown isolation"):
        SandboxExecutor(isolation="container")  # type: ignore[arg-type]
