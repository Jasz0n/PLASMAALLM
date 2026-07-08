"""Model-backed student with study memory.

Design decisions
----------------
- A :class:`ModelStudent` wraps any :class:`LanguageModel` (echo in
  tests, Hugging Face in real runs) — the teacher cannot tell the
  difference, per the Student protocol.
- Studying is *memory-augmented generation*: studied facts live in a
  bounded note store. On ``solve`` the student first tries an exact
  retrieval hit (answering from memory with high confidence), otherwise
  it queries the model with its most relevant notes in the prompt.
  This makes learning real and measurable today, independent of
  weight-level fine-tuning, which arrives as another Trainer backend.
- Confidence is self-reported (see ``confidence.py``); when the model
  does not comply, a conservative configurable default applies.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from allm.core.logging import get_logger
from allm.exam.base import Answer, Question
from allm.exam.grading import normalise
from allm.models.base import LanguageModel, LogProbModel
from allm.students.base import student_types
from allm.students.confidence import CONFIDENCE_INSTRUCTION, parse_confidence
from allm.students.logprob_confidence import estimate_from_logprobs

logger = get_logger("students.model")


class ModelStudentConfig(BaseModel):
    """Tunables for a model-backed student."""

    model_config = ConfigDict(frozen=True)

    max_notes: int = Field(default=64, ge=1)
    notes_in_prompt: int = Field(default=8, ge=0)
    max_pinned_notes: int = Field(default=32, ge=0)
    default_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    memory_confidence: float = Field(default=0.95, ge=0.0, le=1.0)


@student_types.register("model")
class ModelStudent:
    """A student that answers with a language model plus studied notes."""

    def __init__(
        self,
        student_id: str,
        specialty: str,
        model: LanguageModel,
        config: ModelStudentConfig | None = None,
    ) -> None:
        self._id = student_id
        self._specialty = specialty
        self._model = model
        self._config = config or ModelStudentConfig()
        self._notes: dict[str, tuple[str, str]] = {}  # normalised q -> (q, a)
        self._pinned: set[str] = set()
        self._active_adapter_id: str | None = None

    @property
    def active_adapter_id(self) -> str | None:
        """LoRA adapter that produced answers, when weight-trained."""
        return self._active_adapter_id

    def set_adapter(self, adapter_id: str | None) -> None:
        """Record which adapter is loaded after a LoRA fine-tune."""
        self._active_adapter_id = adapter_id

    @property
    def student_id(self) -> str:
        return self._id

    @property
    def specialty(self) -> str:
        return self._specialty

    @property
    def notes(self) -> list[tuple[str, str]]:
        """Studied (question, answer) pairs, oldest first."""
        return list(self._notes.values())

    def snapshot_notes(self) -> dict[str, tuple[str, str]]:
        """Copy of the note store for temporary clearing during eval."""
        return dict(self._notes)

    def replace_notes(self, notes: dict[str, tuple[str, str]]) -> None:
        """Restore or clear studied notes (used for weight-only held-out exams)."""
        self._notes = dict(notes)
        self._pinned = {key for key in self._notes if key in self._pinned}

    def pinned_note_count(self) -> int:
        """Number of notes protected from eviction."""
        return len(self._pinned)

    def study(self, prompt: str, answer: str, *, pinned: bool = False) -> None:
        """Memorise one fact; oldest non-pinned notes drop beyond max_notes."""
        key = normalise(prompt)
        self._notes.pop(key, None)
        self._notes[key] = (prompt, answer)
        if pinned:
            self._pinned.add(key)
        while len(self._notes) > self._config.max_notes:
            evicted = self._evict_one_note()
            if evicted is None:
                break

    def _evict_one_note(self) -> str | None:
        """Drop the oldest evictable note, or None when every note is pinned."""
        for key in list(self._notes):
            if key not in self._pinned:
                self._notes.pop(key)
                return key
        if len(self._pinned) <= self._config.max_pinned_notes:
            return None
        oldest = next(iter(self._notes))
        self._pinned.discard(oldest)
        dropped = self._notes.pop(oldest)
        logger.debug("%s forgot pinned note %r", self._id, dropped[0])
        return oldest

    def build_prompt(self, question: Question) -> str:
        """The exact prompt sent to the model (public for tests/inspection)."""
        lines = [
            f"You are a student specialising in {self._specialty}.",
            "Answer the question concisely.",
            CONFIDENCE_INSTRUCTION,
        ]
        recent = self._notes_for_prompt()
        if recent:
            lines.append("Facts you have studied:")
            lines.extend(f"- Q: {q} A: {a}" for q, a in recent)
        lines.append(f"Question: {question.prompt}")
        lines.append("Answer:")
        return "\n".join(lines)

    def _notes_for_prompt(self) -> list[tuple[str, str]]:
        """Pinned notes first, then most recent evictable notes."""
        rows: list[tuple[str, str]] = []
        for key in self._notes:
            if key in self._pinned:
                rows.append(self._notes[key])
        for key in reversed(list(self._notes)):
            if key in self._pinned:
                continue
            rows.append(self._notes[key])
        return rows[-self._config.notes_in_prompt :]

    def solve(self, question: Question) -> Answer:
        remembered = self._notes.get(normalise(question.prompt))
        if remembered is not None:
            return Answer(
                question_id=question.id,
                text=remembered[1],
                confidence=self._config.memory_confidence,
                reasoning="retrieved from studied notes",
            )
        prompt = self.build_prompt(question)
        if isinstance(self._model, LogProbModel):
            raw, logprobs = self._model.generate_with_logprobs(prompt)
            text, self_conf = parse_confidence(raw)
            lp_conf = estimate_from_logprobs(logprobs)
            confidence = (
                lp_conf
                if lp_conf is not None
                else self_conf
                if self_conf is not None
                else self._config.default_confidence
            )
            return Answer(
                question_id=question.id,
                text=text,
                confidence=confidence,
                logprob_confidence=lp_conf,
                self_reported_confidence=self_conf,
            )
        raw = self._model.generate(prompt)
        text, self_conf = parse_confidence(raw)
        confidence = self_conf if self_conf is not None else self._config.default_confidence
        return Answer(
            question_id=question.id,
            text=text,
            confidence=confidence,
            self_reported_confidence=self_conf,
        )
