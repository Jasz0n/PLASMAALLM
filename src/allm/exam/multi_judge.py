"""Multi-dimensional LLM grading — curriculum, alignment, evidence (M44)."""

from __future__ import annotations

import os
import re

from allm.core.logging import get_logger
from allm.exam.base import Answer, Question, QuestionResult
from allm.exam.grading import ExactMatchGrader, graders
from allm.exam.verdicts import AlignmentLevel, MultiDimensionalVerdict
from allm.models.base import LanguageModel

logger = get_logger("exam.multi_judge")

_CURRICULUM = re.compile(
    r"^\s*CURRICULUM_VERDICT:\s*(correct|incorrect)\s*$", re.IGNORECASE | re.MULTILINE
)
_CURRICULUM_SCORE = re.compile(
    r"^\s*CURRICULUM_SCORE:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE
)
_CURRICULUM_REASON = re.compile(
    r"^\s*CURRICULUM_REASON:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_ALIGNMENT = re.compile(
    r"^\s*ALIGNMENT:\s*(aligned|partially_aligned|disputed|unsupported|unknown)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ALIGNMENT_SCORE = re.compile(
    r"^\s*ALIGNMENT_SCORE:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE
)
_ALIGNMENT_REASON = re.compile(
    r"^\s*ALIGNMENT_REASON:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_EVIDENCE_SCORE = re.compile(
    r"^\s*EVIDENCE_SCORE:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE
)
_EVIDENCE_REASON = re.compile(
    r"^\s*EVIDENCE_REASON:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)


def multi_judge_enabled() -> bool:
    return os.environ.get("ALLM_MULTI_JUDGE", "1") == "1"


def build_multi_judge_prompt(
    question: Question,
    answer: Answer,
    *,
    source_excerpt: str | None = None,
) -> str:
    """Prompt for curriculum-bound grading plus alignment and evidence."""
    lines = [
        "You are an independent exam evaluator with three separate roles.",
        "",
        "1) CURRICULUM JUDGE — Did the student learn what the teacher intended?",
        "   Use ONLY the reference answer and any supplied source excerpt.",
        "   Do NOT use outside or mainstream scientific knowledge for this role.",
        "   Accept paraphrases when the meaning matches the reference.",
        "",
        "2) SCIENTIFIC ALIGNMENT — How does the answer compare with mainstream consensus?",
        "   This is separate from curriculum correctness.",
        "",
        "3) EVIDENCE JUDGE — How well is the claim supported by the supplied material?",
        "",
        f"Question: {question.prompt}",
    ]
    if question.expected is not None:
        lines.append(f"Reference answer: {question.expected}")
    if source_excerpt:
        lines.append(f"Source excerpt: {source_excerpt}")
    lines.extend(
        [
            f"Student answer: {answer.text}",
            "",
            "Reply with exactly these lines and nothing else:",
            "CURRICULUM_VERDICT: correct|incorrect",
            "CURRICULUM_SCORE: 0.0-1.0",
            "CURRICULUM_REASON: one short sentence",
            "ALIGNMENT: aligned|partially_aligned|disputed|unsupported|unknown",
            "ALIGNMENT_SCORE: 0.0-1.0",
            "ALIGNMENT_REASON: one short sentence",
            "EVIDENCE_SCORE: 0.0-1.0",
            "EVIDENCE_REASON: one short sentence",
        ]
    )
    return "\n".join(lines)


def parse_multi_judge_response(text: str) -> MultiDimensionalVerdict:
    """Parse structured multi-judge output."""
    curriculum_match = _CURRICULUM.search(text)
    curriculum_correct = (
        curriculum_match is not None and curriculum_match.group(1).lower() == "correct"
    )
    curriculum_score_match = _CURRICULUM_SCORE.search(text)
    curriculum_score = (
        max(0.0, min(1.0, float(curriculum_score_match.group(1))))
        if curriculum_score_match
        else (1.0 if curriculum_correct else 0.0)
    )
    curriculum_reason = (
        _CURRICULUM_REASON.search(text).group(1).strip()
        if _CURRICULUM_REASON.search(text)
        else None
    )

    alignment_match = _ALIGNMENT.search(text)
    alignment: AlignmentLevel = "unknown"
    if alignment_match is not None:
        alignment = alignment_match.group(1).lower()  # type: ignore[assignment]
    alignment_score_match = _ALIGNMENT_SCORE.search(text)
    alignment_score = (
        max(0.0, min(1.0, float(alignment_score_match.group(1))))
        if alignment_score_match
        else MultiDimensionalVerdict.alignment_to_score(alignment)
    )
    alignment_reason = (
        _ALIGNMENT_REASON.search(text).group(1).strip()
        if _ALIGNMENT_REASON.search(text)
        else None
    )

    evidence_score_match = _EVIDENCE_SCORE.search(text)
    evidence_score = (
        max(0.0, min(1.0, float(evidence_score_match.group(1))))
        if evidence_score_match
        else 0.5
    )
    evidence_reason = (
        _EVIDENCE_REASON.search(text).group(1).strip()
        if _EVIDENCE_REASON.search(text)
        else None
    )

    return MultiDimensionalVerdict(
        curriculum_correct=curriculum_correct,
        curriculum_score=round(curriculum_score, 4),
        curriculum_reason=curriculum_reason,
        alignment=alignment,
        alignment_score=round(alignment_score, 4),
        alignment_reason=alignment_reason,
        evidence_score=round(evidence_score, 4),
        evidence_reason=evidence_reason,
    )


@graders.register("multi_judge")
class MultiDimensionalGrader:
    """Grades with curriculum-bound verdict plus alignment and evidence."""

    def __init__(
        self,
        model: LanguageModel,
        *,
        compare_exact: bool = True,
        source_excerpt: str | None = None,
    ) -> None:
        self._model = model
        self._exact = ExactMatchGrader("contains") if compare_exact else None
        self._source_excerpt = source_excerpt

    def grade(self, question: Question, answer: Answer) -> QuestionResult:
        prompt = build_multi_judge_prompt(
            question,
            answer,
            source_excerpt=self._source_excerpt,
        )
        raw = self._model.generate(prompt)
        verdict = parse_multi_judge_response(raw)
        exact_correct: bool | None = None
        if self._exact is not None and question.expected is not None:
            exact = self._exact.grade(question, answer)
            exact_correct = exact.correct
            if exact.correct != verdict.curriculum_correct:
                logger.info(
                    "multi-judge disagrees with exact_match on %s: "
                    "curriculum=%s exact=%s alignment=%s evidence=%.2f",
                    question.id,
                    verdict.curriculum_correct,
                    exact.correct,
                    verdict.alignment,
                    verdict.evidence_score,
                )
        verdict = verdict.model_copy(update={"exact_match": exact_correct})
        feedback = self._format_feedback(verdict)
        return QuestionResult(
            question=question,
            answer=answer,
            score=verdict.curriculum_score,
            correct=verdict.curriculum_correct,
            feedback=feedback,
            verdict=verdict,
        )

    @staticmethod
    def _format_feedback(verdict: MultiDimensionalVerdict) -> str:
        parts = [
            f"curriculum={verdict.curriculum_score:.2f}",
            f"alignment={verdict.alignment}({verdict.alignment_score:.2f})",
            f"evidence={verdict.evidence_score:.2f}",
        ]
        if verdict.curriculum_reason:
            parts.append(verdict.curriculum_reason)
        if verdict.exact_match is not None and verdict.exact_match != verdict.curriculum_correct:
            parts.insert(0, "curriculum/exact disagree")
        return "; ".join(parts)
