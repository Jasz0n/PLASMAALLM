"""Student interface.

Design decisions
----------------
- The teacher only ever sees this protocol: identity, specialty, and
  the ability to answer questions with self-reported confidence.
  Model-backed students, fine-tuning and failure memory arrive in
  Phase 3 behind the same surface; the teacher will not change.
- Students never receive a reference to the teacher — the dependency
  is strictly teacher -> student, which structurally enforces
  "students should never directly modify the Teacher".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from allm.core.registry import Registry
from allm.exam.base import Answer, Question


@runtime_checkable
class Student(Protocol):
    """Anything that can sit an exam."""

    @property
    def student_id(self) -> str: ...

    @property
    def specialty(self) -> str:
        """Domain this student focuses on (Plan.md: one area each)."""
        ...

    def solve(self, question: Question) -> Answer: ...


student_types: Registry[type] = Registry("student_type")
