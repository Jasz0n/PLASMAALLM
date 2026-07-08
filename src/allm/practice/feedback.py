"""Review outcomes become learning signal (Roadmap M49).

A human's verdict on a contribution is a graded exam: approval is a
pass, rejection is a failure — *with the reviewer's reason attached as
feedback*, because the reason is the teaching. Outcomes fold into the
same teacher state every exam uses, so KEL's learning gain sees review
history exactly like any other measured knowledge, and rejections land
in the ``FailureLog`` next to every other mistake the student learns
from.
"""

from __future__ import annotations

from allm.core.logging import get_logger
from allm.exam.base import Answer, ExamResult, Question, QuestionResult
from allm.practice.contribution import Contribution
from allm.students.failures import FailureLog
from allm.teacher.state import KnowledgeState

logger = get_logger("practice.feedback")


def contribution_question(contribution: Contribution, *, topic: str) -> Question:
    """The task the apprentice actually faced, as a gradable question."""
    return Question(
        id=f"{contribution.id}-review",
        prompt=(
            f"Propose the complete corrected content of "
            f"{contribution.patch.file!r} so that "
            f"{contribution.test_selector!r} passes."
        ),
        # Approval certifies the patch as a correct answer, studyable
        # later; a rejected patch leaves the right answer unknown.
        expected=(
            contribution.patch.content
            if contribution.status in ("approved", "applied")
            else None
        ),
        topic=topic,
        kind="practice",
    )


def review_exam_result(contribution: Contribution, *, topic: str) -> ExamResult:
    """One review verdict as a one-question graded exam."""
    if contribution.status not in ("approved", "rejected", "applied"):
        raise ValueError(
            f"contribution {contribution.id} is {contribution.status!r} — "
            "only reviewed contributions carry a learning signal"
        )
    passed = contribution.status in ("approved", "applied")
    question = contribution_question(contribution, topic=topic)
    graded = QuestionResult(
        question=question,
        answer=Answer(
            question_id=question.id,
            text=contribution.patch.content,
            confidence=1.0 if contribution.trial_outcome == "pass" else 0.5,
            reasoning=contribution.patch.reasoning,
        ),
        score=1.0 if passed else 0.0,
        correct=passed,
        feedback=contribution.review_reason,
    )
    return ExamResult(
        exam_id=f"review-{contribution.id}",
        student_id=contribution.patch.author,
        results=(graded,),
    )


def record_review_outcome(
    contribution: Contribution,
    *,
    state: KnowledgeState,
    failures: FailureLog | None = None,
    topic: str,
    smoothing: float = 0.3,
) -> ExamResult:
    """Fold one review verdict into teacher state (and failures on reject).

    Returns the recorded exam result. Confidence on ``topic`` moves the
    same way any exam moves it; a rejection also lands in the failure
    log with the reviewer's reason as feedback.
    """
    result = review_exam_result(contribution, topic=topic)
    state.record_exam_result(result, smoothing)
    if failures is not None:
        for failed in result.failures():
            failures.record(result.student_id, failed)
    logger.info(
        "review outcome for %s: %s -> %s confidence on %r",
        contribution.id,
        contribution.status,
        f"{state.confidence(result.student_id, topic):.2f}",
        topic,
    )
    return result
