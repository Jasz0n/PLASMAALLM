"""The Teacher: evaluates students, assigns goals, measures progress.

Design decisions
----------------
- The teacher is pure orchestration: exam creation is delegated to an
  injected :class:`ExamGenerator`, scoring to an injected
  :class:`Grader`, persistence to :class:`KnowledgeState`. Each can be
  swapped independently (SOLID: the teacher has one reason to change).
- Every public method both returns its result *and* records it in the
  knowledge state, so the teacher's world view is always reconstructable
  from storage — including full history, because storage is versioned.
- The teacher holds no reference-able mutable state of its own and
  hands students nothing but questions; students cannot reach the
  teacher or its records.
"""

from __future__ import annotations

import os
from typing import Sequence

from allm.core.logging import get_logger
from allm.exam.base import Exam, ExamGenerator, ExamResult
from allm.exam.grading import Grader
from allm.students.base import Student
from allm.students.model_student import ModelStudent
from allm.teacher.state import KnowledgeState
from allm.teacher.types import LearningGoal, ProgressReport, TeacherConfig, TopicProgress

logger = get_logger("teacher")


class Teacher:
    """Coordinates evaluation and goal-setting for all students."""

    def __init__(
        self,
        state: KnowledgeState,
        exam_generator: ExamGenerator,
        grader: Grader,
        config: TeacherConfig | None = None,
    ) -> None:
        self._state = state
        self._generator = exam_generator
        self._grader = grader
        self._config = config or TeacherConfig()

    @property
    def state(self) -> KnowledgeState:
        """Read access to the global knowledge state."""
        return self._state

    @property
    def grader(self) -> Grader:
        """Grader used for exams and mediated consultation validation."""
        return self._grader

    def set_exam_paraphrase(self, enabled: bool) -> None:
        """Toggle paraphrase mode when the generator supports it."""
        if hasattr(self._generator, "paraphrase"):
            self._generator.paraphrase = enabled

    def create_exam(
        self,
        *,
        topics: Sequence[str] | None = None,
        num_questions: int = 10,
        seed: int | None = None,
    ) -> Exam:
        """Create an exam via the injected generator."""
        exam = self._generator.generate(
            topics=topics, num_questions=num_questions, seed=seed
        )
        logger.info("created %s with %d questions", exam.id, len(exam.questions))
        return exam

    def evaluate(self, student: Student, exam: Exam) -> ExamResult:
        """Sit ``student`` through ``exam``, grade it, record everything."""
        graded = tuple(
            self._grader.grade(question, student.solve(question))
            for question in exam.questions
        )
        result = ExamResult(
            exam_id=exam.id,
            student_id=student.student_id,
            results=graded,
            adapter_id=student.active_adapter_id
            if isinstance(student, ModelStudent)
            else None,
        )
        self._state.record_exam_result(result, self._config.confidence_smoothing)
        log_path = os.environ.get("ALLM_JUDGE_DISAGREEMENT_LOG")
        if log_path:
            from allm.exam.disagreement_log import append_disagreements

            appended = append_disagreements(log_path, result)
            if appended:
                logger.debug("logged %d judge disagreement(s) to %s", appended, log_path)
        logger.info(
            "%s scored %.2f on %s (%d/%d correct)",
            student.student_id,
            result.score,
            exam.id,
            sum(r.correct for r in graded),
            len(graded),
        )
        return result

    def assign_goals(self, student_id: str) -> list[LearningGoal]:
        """Turn the student's weakest topics into prioritised goals.

        Phase 2 heuristic: any topic below the weakness threshold, worst
        first, priority = 1 - confidence. Phase 4's planner will supply
        richer prioritisation through the same call site.
        """
        weak = [
            (topic, confidence)
            for topic in self._state.topics(student_id)
            if (confidence := self._state.confidence(student_id, topic)) is not None
            and confidence < self._config.weakness_threshold
        ]
        weak.sort(key=lambda item: item[1])
        goals = [
            LearningGoal(
                student_id=student_id,
                topic=topic,
                priority=round(1.0 - confidence, 4),
                reason=(
                    f"confidence {confidence:.2f} below threshold "
                    f"{self._config.weakness_threshold:.2f}"
                ),
            )
            for topic, confidence in weak[: self._config.max_goals]
        ]
        self._state.record_goals(student_id, goals)
        logger.info("assigned %d goal(s) to %s", len(goals), student_id)
        return goals

    def progress(self, student_id: str) -> ProgressReport:
        """Summarise a student's trajectory from recorded history."""
        exams = self._state.exam_results(student_id)
        topics = []
        for topic in self._state.topics(student_id):
            history = self._state.confidence_history(student_id, topic)
            if history:
                topics.append(
                    TopicProgress(
                        topic=topic,
                        first=history[0][1],
                        latest=history[-1][1],
                        observations=len(history),
                    )
                )
        mean_score = (
            sum(e.score for e in exams) / len(exams) if exams else 0.0
        )
        return ProgressReport(
            student_id=student_id,
            exams_taken=len(exams),
            mean_score=mean_score,
            topics=tuple(topics),
        )
