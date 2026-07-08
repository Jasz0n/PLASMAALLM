"""Exam generation from datasets.

Builds exams by sampling labelled examples (``Sample`` with a target)
from any dataset loader. Reproducible via ``seed``. This gives the
teacher real exams today; generative exam creation is Phase 7.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from allm.data.base import Sample
from allm.exam.base import Exam, Question, exam_generators


@exam_generators.register("dataset")
class DatasetExamGenerator:
    """Samples questions from a pool of labelled dataset samples.

    A sample maps to a question as: input -> prompt, target -> expected,
    ``metadata["topic"]`` -> topic (default "general"),
    ``metadata["kind"]`` -> kind (default "factual"). Samples without a
    target are rejected up front: an exam question we cannot grade is a
    silent hole in evaluation.
    """

    def __init__(self, samples: Iterable[Sample]) -> None:
        self._samples = list(samples)
        unlabelled = [s.id for s in self._samples if s.target is None]
        if unlabelled:
            raise ValueError(
                f"samples without targets cannot become exam questions: {unlabelled[:5]}"
            )
        if not self._samples:
            raise ValueError("cannot generate exams from an empty sample pool")
        self._counter = 0

    def topics(self) -> list[str]:
        """Distinct topics available in the pool, sorted."""
        return sorted({self._topic(s) for s in self._samples})

    def generate(
        self,
        *,
        topics: Sequence[str] | None = None,
        num_questions: int = 10,
        seed: int | None = None,
    ) -> Exam:
        pool = self._samples
        if topics is not None:
            wanted = set(topics)
            pool = [s for s in pool if self._topic(s) in wanted]
            if not pool:
                raise ValueError(f"no samples for topics {sorted(wanted)}; have {self.topics()}")
        rng = random.Random(seed)
        chosen = rng.sample(pool, k=min(num_questions, len(pool)))
        self._counter += 1
        exam_id = f"exam-{self._counter:04d}"
        return Exam(
            id=exam_id,
            title=f"{exam_id} ({', '.join(topics) if topics else 'all topics'})",
            questions=tuple(
                Question(
                    id=f"{exam_id}-q{i}",
                    prompt=s.input,
                    expected=s.target,
                    topic=self._topic(s),
                    kind=s.metadata.get("kind", "factual"),
                )
                for i, s in enumerate(chosen, start=1)
            ),
        )

    @staticmethod
    def _topic(sample: Sample) -> str:
        return str(sample.metadata.get("topic", "general"))
