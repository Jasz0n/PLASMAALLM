"""Episodic memory over the versioned record store.

Design decisions
----------------
- Episodes are append-only records in namespace ``episodes`` (an id
  collision creates a new version rather than overwriting — Plan.md's
  versioning rule holds here as everywhere).
- Recall is filter-based plus a dependency-free lexical search (token
  overlap). A vector backend (FAISS/Chroma) can register as another
  ``memory_backends`` entry later; the recall/search interface is the
  contract, not the index.
- ``remember_exam`` is the standard bridge from graded exams to
  memory: one success/failure episode per question, carrying the
  student's reasoning trace when present.
"""

from __future__ import annotations

import json
import re
from itertools import count
from typing import Iterable

from allm.core.registry import Registry
from allm.exam.base import ExamResult
from allm.memory.types import Episode, EpisodeKind
from allm.storage.base import RecordStore

memory_backends: Registry[type] = Registry("memory_backend")

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@memory_backends.register("episodic")
class EpisodicMemory:
    """Append-only episode log with filtered recall and lexical search."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store
        self._sequence = count(len(store.keys("episodes")) + 1)

    def append(self, episode: Episode) -> Episode:
        self._store.put(
            "episodes",
            episode.id,
            json.loads(episode.model_dump_json()),
            reason=f"{episode.kind} for {episode.actor}",
        )
        return episode

    def remember(
        self,
        actor: str,
        kind: EpisodeKind,
        summary: str,
        *,
        topic: str = "general",
        detail: dict | None = None,
        confidence: float | None = None,
    ) -> Episode:
        """Convenience: build an id-stamped episode and append it."""
        episode = Episode(
            id=f"ep-{next(self._sequence):06d}",
            actor=actor,
            kind=kind,
            topic=topic,
            summary=summary,
            detail=detail or {},
            confidence=confidence,
        )
        return self.append(episode)

    def remember_exam(self, result: ExamResult) -> list[Episode]:
        """Record one success/failure episode per graded question."""
        episodes = []
        for graded in result.results:
            kind: EpisodeKind = "success" if graded.correct else "failure"
            episodes.append(
                self.remember(
                    result.student_id,
                    kind,
                    f"{'answered' if graded.correct else 'missed'} "
                    f"{graded.question.prompt!r}",
                    topic=graded.question.topic,
                    detail={
                        "exam_id": result.exam_id,
                        "question_id": graded.question.id,
                        "prompt": graded.question.prompt,
                        "given": graded.answer.text,
                        "expected": graded.question.expected,
                        "reasoning": graded.answer.reasoning,
                        "score": graded.score,
                    },
                    confidence=graded.answer.confidence,
                )
            )
        return episodes

    def recall(
        self,
        *,
        actor: str | None = None,
        kind: EpisodeKind | None = None,
        topic: str | None = None,
        limit: int | None = None,
    ) -> list[Episode]:
        """Episodes matching all given filters, oldest first."""
        matches = [
            e
            for e in self._all()
            if (actor is None or e.actor == actor)
            and (kind is None or e.kind == kind)
            and (topic is None or e.topic == topic)
        ]
        matches.sort(key=lambda e: (e.occurred_at, e.id))
        return matches if limit is None else matches[-limit:]

    def search(self, query: str, *, limit: int = 5) -> list[Episode]:
        """Lexical search over summaries, best token overlap first."""
        wanted = _tokens(query)
        if not wanted:
            return []
        scored = []
        for episode in self._all():
            overlap = len(wanted & _tokens(episode.summary))
            if overlap:
                scored.append((overlap, episode))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [episode for _, episode in scored[:limit]]

    def _all(self) -> Iterable[Episode]:
        for key in self._store.keys("episodes"):
            record = self._store.get("episodes", key)
            yield Episode.model_validate(record.value)
