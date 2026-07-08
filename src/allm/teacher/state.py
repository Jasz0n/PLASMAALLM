"""Global knowledge state — the teacher's persistent view of the system.

A thin, typed layer over the versioned :class:`RecordStore`. Namespaces:

    exam_results       key "<student>::<exam>"   full graded exams
    topic_confidence   key "<student>::<topic>"  EMA-updated confidence
    goals              key "<student>"           latest assigned goals

Because the store is append-only, confidence *history* (Plan.md:
"previous belief, new belief, reason for change") requires no extra
machinery — it is simply the record history.
"""

from __future__ import annotations

import json
from datetime import datetime

from allm.exam.base import ExamResult
from allm.storage.base import RecordStore
from allm.teacher.types import LearningGoal

SEP = "::"


class KnowledgeState:
    """Typed persistence for exam results, confidences and goals."""

    def __init__(self, store: RecordStore) -> None:
        self._store = store

    # -- exam results -------------------------------------------------

    def record_exam_result(self, result: ExamResult, smoothing: float) -> None:
        """Store a graded exam and fold its topic scores into confidence.

        ``smoothing`` in (0, 1]: weight of this exam's score against the
        previous confidence (exponential moving average).
        """
        self._store.put(
            "exam_results",
            f"{result.student_id}{SEP}{result.exam_id}",
            json.loads(result.model_dump_json()),
            reason=f"exam {result.exam_id}, score {result.score:.2f}",
        )
        for topic, score in result.topic_scores().items():
            previous = self.confidence(result.student_id, topic)
            updated = score if previous is None else (
                smoothing * score + (1 - smoothing) * previous
            )
            self._store.put(
                "topic_confidence",
                f"{result.student_id}{SEP}{topic}",
                {"confidence": updated, "exam_score": score},
                reason=f"exam {result.exam_id}: topic scored {score:.2f}",
            )

    def exam_results(self, student_id: str) -> list[ExamResult]:
        """All recorded exams for a student, oldest first."""
        results = []
        for key in self._store.keys("exam_results"):
            sid, _, _ = key.partition(SEP)
            if sid == student_id:
                record = self._store.get("exam_results", key)
                results.append(ExamResult.model_validate(record.value))
        results.sort(key=lambda r: r.taken_at)
        return results

    # -- confidence ----------------------------------------------------

    def confidence(self, student_id: str, topic: str) -> float | None:
        record = self._store.get("topic_confidence", f"{student_id}{SEP}{topic}")
        return None if record is None else float(record.value["confidence"])

    def confidence_history(
        self, student_id: str, topic: str
    ) -> list[tuple[datetime, float]]:
        """(timestamp, confidence) for every belief revision, oldest first."""
        history = self._store.history("topic_confidence", f"{student_id}{SEP}{topic}")
        return [(r.created_at, float(r.value["confidence"])) for r in history]

    def topics(self, student_id: str) -> list[str]:
        """Topics this student has ever been examined on."""
        result = []
        for key in self._store.keys("topic_confidence"):
            sid, _, topic = key.partition(SEP)
            if sid == student_id:
                result.append(topic)
        return sorted(result)

    def students(self) -> list[str]:
        """Every student that has confidence records."""
        seen = {key.partition(SEP)[0] for key in self._store.keys("topic_confidence")}
        return sorted(seen)

    def global_confidence(self, topic: str) -> float | None:
        """Mean of all students' latest confidence on ``topic``."""
        values = [
            c
            for sid in self.students()
            if (c := self.confidence(sid, topic)) is not None
        ]
        return sum(values) / len(values) if values else None

    # -- goals ---------------------------------------------------------

    def record_goals(self, student_id: str, goals: list[LearningGoal]) -> None:
        self._store.put(
            "goals",
            student_id,
            {"goals": [json.loads(g.model_dump_json()) for g in goals]},
            reason=f"assigned {len(goals)} goal(s)",
        )

    def current_goals(self, student_id: str) -> list[LearningGoal]:
        record = self._store.get("goals", student_id)
        if record is None:
            return []
        return [LearningGoal.model_validate(g) for g in record.value["goals"]]
