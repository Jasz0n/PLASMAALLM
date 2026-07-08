"""Exam generation with paraphrased prompts (same expected answers)."""

from __future__ import annotations

import random
from typing import Sequence

from allm.data.base import Sample
from allm.exam.base import Exam, Question, exam_generators
from allm.exam.dataset_generator import DatasetExamGenerator
from allm.exam.paraphrase import paraphrase_definition_prompt


@exam_generators.register("paraphrase")
class ParaphraseExamGenerator:
    """Like :class:`DatasetExamGenerator` but rewrites prompts for held-out tests."""

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        paraphrase: bool = True,
    ) -> None:
        self._inner = DatasetExamGenerator(samples)
        self._paraphrase = paraphrase

    def topics(self) -> list[str]:
        return self._inner.topics()

    @property
    def paraphrase(self) -> bool:
        return self._paraphrase

    @paraphrase.setter
    def paraphrase(self, enabled: bool) -> None:
        self._paraphrase = enabled

    def generate(
        self,
        *,
        topics: Sequence[str] | None = None,
        num_questions: int = 10,
        seed: int | None = None,
    ) -> Exam:
        exam = self._inner.generate(topics=topics, num_questions=num_questions, seed=seed)
        if not self._paraphrase:
            return exam
        rng = random.Random(seed)
        questions: list[Question] = []
        for index, question in enumerate(exam.questions, start=1):
            variant = rng.randint(0, 2) if seed is None else (seed + index) % 3
            prompt = paraphrase_definition_prompt(question.prompt, variant=variant)
            questions.append(question.model_copy(update={"prompt": prompt}))
        return exam.model_copy(update={"questions": tuple(questions)})
