"""Kernel-enforced execution limits (Roadmap M50), offline."""

import sys

import pytest

from allm.exam import CodingGrader
from allm.exam.base import Answer, Question
from allm.practice import (
    PracticeProcedure,
    ResourceLimits,
    SandboxExecutor,
    VariableSpec,
)

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="rlimits are POSIX-only"
)

TIGHT = ResourceLimits(
    cpu_seconds=2, memory_bytes=128 * 1024 * 1024, file_size_bytes=1024 * 1024
)


def procedure(program: str, **kwargs) -> PracticeProcedure:
    return PracticeProcedure(
        id="limited",
        description="resource limit probe",
        program=program,
        variables=(VariableSpec(name="n", default=1),),
        limits=TIGHT,
        **kwargs,
    )


@posix_only
def test_memory_bomb_is_stopped_by_the_kernel() -> None:
    run = SandboxExecutor().run(
        procedure("data = bytearray(512 * 1024 * 1024)\nprint('survived')\n")
    )
    assert run.status == "crash"
    assert "survived" not in run.outcome


@posix_only
def test_cpu_spin_is_stopped_even_with_a_generous_wall_timeout() -> None:
    # wall timeout of 30s would let a spin burn half a minute of CPU;
    # RLIMIT_CPU=2 kills it long before that.
    run = SandboxExecutor().run(
        procedure("while True:\n    pass\n", timeout_seconds=30.0)
    )
    assert run.status in ("crash", "timeout")
    assert run.duration_seconds < 10


@posix_only
def test_disk_filling_is_stopped() -> None:
    run = SandboxExecutor().run(
        procedure(
            "import tempfile\n"
            "f = tempfile.TemporaryFile()\n"
            "f.write(b'x' * (8 * 1024 * 1024))\n"
            "print('survived')\n"
        )
    )
    assert run.status == "crash"
    assert "survived" not in run.outcome


def test_wellbehaved_code_is_untouched() -> None:
    run = SandboxExecutor().run(procedure("print(sum(range(100)))\n"))
    assert run.status == "ok" and run.outcome == "4950"


@posix_only
def test_coding_grader_shares_the_guard() -> None:
    grader = CodingGrader(timeout_seconds=10.0, limits=TIGHT)
    question = Question(id="q1", prompt="allocate", expected="done", kind="coding")
    bomb = Answer(
        question_id="q1",
        text="```python\ndata = bytearray(512 * 1024 * 1024)\nprint('done')\n```",
        confidence=0.9,
    )
    result = grader.grade(question, bomb)
    assert not result.correct
    assert "crashed" in (result.feedback or "")


def test_limits_validate_sane_floors() -> None:
    with pytest.raises(ValueError):
        ResourceLimits(memory_bytes=1024)  # below the 32MB interpreter floor
    with pytest.raises(ValueError):
        ResourceLimits(cpu_seconds=0)
