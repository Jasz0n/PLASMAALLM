"""Coding grader: run the submission, compare its output.

The student's answer is a Python program; ``expected`` is the exact
stdout it must produce. Programs run in a subprocess with a hard
timeout, no shell, and kernel resource limits (CPU, memory, file size —
M50): the kernel stops what the timeout alone cannot.

Research-platform caveat (documented, deliberate): submissions still
run with the experimenter's own privileges — rlimits are the floor;
full OS isolation (container/jail) before grading untrusted third-party
submissions remains the M50 exit bar.
"""

from __future__ import annotations

import re
import subprocess
import sys

from allm.core.logging import get_logger
from allm.exam.base import Answer, Question, QuestionResult
from allm.exam.grading import graders, normalise

logger = get_logger("exam.coding")

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the first fenced code block, or the raw text if none."""
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


@graders.register("coding")
class CodingGrader:
    """Executes Python answers and checks their stdout."""

    def __init__(
        self, timeout_seconds: float = 5.0, limits: "ResourceLimits | None" = None
    ) -> None:
        from allm.practice.limits import ResourceLimits, clean_env

        self._timeout = timeout_seconds
        self._limits = limits or ResourceLimits()
        self._env = clean_env()

    def grade(self, question: Question, answer: Answer) -> QuestionResult:
        if question.expected is None:
            return self._result(question, answer, False, "no expected output defined")
        code = extract_code(answer.text)
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                preexec_fn=self._limits.preexec(),
                env=self._env,
            )
        except subprocess.TimeoutExpired:
            return self._result(
                question, answer, False, f"timed out after {self._timeout}s"
            )
        if completed.returncode != 0:
            error = completed.stderr.strip().splitlines()
            return self._result(
                question, answer, False, f"crashed: {error[-1] if error else 'unknown'}"
            )
        correct = normalise(completed.stdout) == normalise(question.expected)
        feedback = None if correct else (
            f"printed {completed.stdout.strip()!r}, expected {question.expected!r}"
        )
        return self._result(question, answer, correct, feedback)

    @staticmethod
    def _result(
        question: Question, answer: Answer, correct: bool, feedback: str | None
    ) -> QuestionResult:
        return QuestionResult(
            question=question,
            answer=answer,
            score=1.0 if correct else 0.0,
            correct=correct,
            feedback=feedback,
        )
