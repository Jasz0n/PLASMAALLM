"""Deterministic student for developing and testing the teacher.

Same rationale as ``allm.models.echo``: teacher logic (evaluation,
goal assignment, progress measurement) must be assertable without
model weights. A ScriptedStudent knows some answers; everything else
gets a low-confidence "I don't know" — which conveniently exercises
the weakness-detection path.
"""

from __future__ import annotations

from allm.exam.base import Answer, Question
from allm.exam.grading import normalise
from allm.students.base import student_types


@student_types.register("scripted")
class ScriptedStudent:
    """Answers from a prompt -> answer mapping.

    Args:
        student_id: Unique id, e.g. ``"student-math"``.
        specialty: Topic this student is meant to be good at.
        knowledge: Mapping of question prompt -> correct answer text.
            Prompts are matched after normalisation.
        confident: Confidence reported for known answers (default 0.9);
            unknown answers always report 0.1.
    """

    def __init__(
        self,
        student_id: str,
        specialty: str,
        knowledge: dict[str, str] | None = None,
        confident: float = 0.9,
    ) -> None:
        self._id = student_id
        self._specialty = specialty
        self._knowledge = {normalise(k): v for k, v in (knowledge or {}).items()}
        self._confident = confident

    @property
    def student_id(self) -> str:
        return self._id

    @property
    def specialty(self) -> str:
        return self._specialty

    def learn(self, prompt: str, answer: str) -> None:
        """Teach this student a new fact (simulates studying)."""
        self._knowledge[normalise(prompt)] = answer

    def solve(self, question: Question) -> Answer:
        known = self._knowledge.get(normalise(question.prompt))
        if known is not None:
            return Answer(
                question_id=question.id, text=known, confidence=self._confident
            )
        return Answer(question_id=question.id, text="I don't know", confidence=0.1)
