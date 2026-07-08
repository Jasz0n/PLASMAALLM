"""Generative exam parsing and generation helpers."""

from __future__ import annotations

import re
from typing import Sequence

from allm.core.logging import get_logger
from allm.exam.base import Exam, Question, QuestionKind, exam_generators
from allm.models.base import LanguageModel

logger = get_logger("exam.generative")

_BLOCK = re.compile(
    r"^T:\s*(?P<topic>.+?)\s*\nQ:\s*(?P<question>.+?)\s*\nA:\s*(?P<answer>.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_INLINE = re.compile(
    r"^T:\s*(?P<topic>.+?)\s*\nQ:\s*(?P<question>.+?)\s+A:\s*(?P<answer>.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_TOPIC_BLOCK = re.compile(r"^T:\s*(?P<topic>.+)$", re.MULTILINE | re.IGNORECASE)
_FENCE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    fenced = _FENCE.findall(text)
    return "\n".join(fenced) if fenced else text


def _parse_topic_block(topic: str, body: str) -> tuple[str, str, str] | None:
    body = body.strip()
    if not body:
        return None
    single = re.match(
        r"^Q:\s*(?P<question>.+?)\s+A:\s*(?P<answer>.+)$",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    if single:
        return topic, single.group("question").strip(), single.group("answer").strip()
    inline = _INLINE.search(f"T: {topic}\n{body}")
    if inline:
        return inline.group("topic", "question", "answer")
    q_match = re.search(r"^Q:\s*(.+)$", body, re.MULTILINE | re.IGNORECASE)
    a_match = re.search(r"^A:\s*(.+)$", body, re.MULTILINE | re.IGNORECASE)
    if q_match and a_match:
        return topic, q_match.group(1).strip(), a_match.group(1).strip()
    return None


def parse_questions(text: str) -> list[tuple[str, str, str]]:
    """Extract (topic, question, answer) triples from model output.

    Tries strict blocks first, then inline Q/A lines, then per-topic
    blocks — small models often deviate from the ideal format.
    """
    cleaned = _strip_fences(text).strip()
    seen: set[str] = set()
    results: list[tuple[str, str, str]] = []

    def add(topic: str, question: str, answer: str) -> None:
        key = question.strip().lower()
        if key not in seen:
            seen.add(key)
            results.append((topic.strip(), question.strip(), answer.strip()))

    for match in _BLOCK.finditer(cleaned):
        add(match["topic"], match["question"], match["answer"])
    for match in _INLINE.finditer(cleaned):
        add(match["topic"], match["question"], match["answer"])
    topic_matches = list(_TOPIC_BLOCK.finditer(cleaned))
    for index, match in enumerate(topic_matches):
        start = match.end()
        end = topic_matches[index + 1].start() if index + 1 < len(topic_matches) else len(cleaned)
        parsed = _parse_topic_block(match.group("topic").strip(), cleaned[start:end])
        if parsed:
            add(*parsed)
    return results


_EXAMPLE = """T: math
Q: What is 2+2?
A: 4
T: geography
Q: Capital of France?
A: Paris"""


@exam_generators.register("model")
class ModelExamGenerator:
    """Asks a language model to write an exam."""

    _counter = 0

    def __init__(
        self,
        model: LanguageModel,
        *,
        kind: QuestionKind = "factual",
        difficulty: int = 1,
        max_attempts: int = 2,
    ) -> None:
        if difficulty < 1:
            raise ValueError("difficulty starts at 1")
        self._model = model
        self._kind = kind
        self._difficulty = difficulty
        self._max_attempts = max(1, max_attempts)

    def build_prompt(self, topics: Sequence[str], num_questions: int) -> str:
        """The exact generation prompt (public for tests/inspection)."""
        style = {
            "factual": "short factual questions with unambiguous one-line answers",
            "reasoning": "questions requiring multi-step reasoning",
            "coding": "small Python tasks; the answer is the exact expected stdout",
            "cross_domain": "questions that each combine at least two of the topics",
        }[self._kind]
        return "\n".join(
            [
                f"Write exactly {num_questions} exam questions: {style}.",
                f"Topics: {', '.join(topics)}.",
                f"Difficulty level: {self._difficulty} (1 = beginner, 5 = expert).",
                "Each answer must be short (one word or number when possible).",
                "Use exactly this format for every question — copy the labels T, Q, A:",
                "T: <topic>",
                "Q: <question>",
                "A: <expected answer>",
                "",
                "Example:",
                _EXAMPLE,
            ]
        )

    def generate(
        self,
        *,
        topics: Sequence[str] | None = None,
        num_questions: int = 10,
        seed: int | None = None,
    ) -> Exam:
        chosen_topics = list(topics) if topics else ["general knowledge"]
        prompt = self.build_prompt(chosen_topics, num_questions)
        parsed: list[tuple[str, str, str]] = []
        raw = ""
        for attempt in range(1, self._max_attempts + 1):
            raw = self._model.generate(prompt)
            parsed = parse_questions(raw)
            if parsed:
                break
            logger.warning(
                "exam generation attempt %d/%d produced no parseable questions",
                attempt,
                self._max_attempts,
            )
            if attempt < self._max_attempts:
                prompt = prompt + "\n\nYour previous output was not parseable. Follow the T/Q/A format exactly."

        if not parsed:
            raise ValueError(
                f"model produced no parseable questions (output started: {raw[:80]!r})"
            )
        if len(parsed) < num_questions:
            logger.warning(
                "asked for %d questions, model produced %d", num_questions, len(parsed)
            )
        ModelExamGenerator._counter += 1
        exam_id = f"gen-{ModelExamGenerator._counter:04d}"
        questions = tuple(
            Question(
                id=f"{exam_id}-q{i}",
                prompt=q,
                expected=a,
                topic=t.strip() or chosen_topics[0],
                kind=self._kind,
            )
            for i, (t, q, a) in enumerate(parsed[:num_questions], start=1)
        )
        return Exam(
            id=exam_id,
            title=f"{exam_id} {self._kind} d{self._difficulty} "
            f"({', '.join(chosen_topics)})",
            questions=questions,
        )
