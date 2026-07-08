"""Collect judge disagreements for review datasets (M44)."""

from __future__ import annotations

import json
from pathlib import Path

from allm.exam.base import ExamResult, QuestionResult


def disagreement_record(result: QuestionResult) -> dict | None:
    """Return a review record when grading dimensions disagree."""
    verdict = getattr(result, "verdict", None)
    if verdict is None:
        return None
    exact = verdict.exact_match
    curriculum = verdict.curriculum_correct
    if exact is None and verdict.alignment == "unknown":
        return None
    if exact == curriculum and verdict.alignment in {"aligned", "unknown"}:
        return None
    return {
        "question_id": result.question.id,
        "prompt": result.question.prompt,
        "expected": result.question.expected,
        "student_answer": result.answer.text,
        "topic": result.question.topic,
        "exact_match": exact,
        "curriculum_correct": curriculum,
        "curriculum_score": verdict.curriculum_score,
        "curriculum_reason": verdict.curriculum_reason,
        "alignment": verdict.alignment,
        "alignment_score": verdict.alignment_score,
        "alignment_reason": verdict.alignment_reason,
        "evidence_score": verdict.evidence_score,
        "evidence_reason": verdict.evidence_reason,
    }


def collect_disagreements(exam_result: ExamResult) -> list[dict]:
    """All disagreement records from one graded exam."""
    rows: list[dict] = []
    for result in exam_result.results:
        record = disagreement_record(result)
        if record is not None:
            rows.append(record)
    return rows


def append_disagreements(
    path: Path | str,
    exam_result: ExamResult,
) -> int:
    """Append disagreement records to a JSONL review file."""
    rows = collect_disagreements(exam_result)
    if not rows:
        return 0
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = {"exam_id": exam_result.exam_id, "student_id": exam_result.student_id, **row}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return len(rows)
