"""Confidence calibration: does stated confidence predict correctness?

Derived from stored exam results — no separate storage. Used as an M1
exit criterion alongside KEL learning gain.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from allm.teacher.state import KnowledgeState

_BUCKET_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class CalibrationBucket(BaseModel):
    """Accuracy within one confidence band."""

    model_config = ConfigDict(frozen=True)

    range_label: str
    count: int
    accuracy: float


class CalibrationReport(BaseModel):
    """Summary of self-reported confidence vs actual correctness."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    n_samples: int
    brier_score: float = Field(description="Mean squared error of confidence vs outcome")
    buckets: tuple[CalibrationBucket, ...]


def _bucket_label(low: float, high: float) -> str:
    if high >= 1.0:
        return f"[{low:.1f}, {high:.1f}]"
    return f"[{low:.1f}, {high:.1f})"


def _collect_pairs(
    state: KnowledgeState,
    student_id: str,
    *,
    confidence_attr: str,
) -> list[tuple[float, bool]]:
    pairs: list[tuple[float, bool]] = []
    for exam in state.exam_results(student_id):
        for qr in exam.results:
            value = getattr(qr.answer, confidence_attr, None)
            if value is not None:
                pairs.append((float(value), qr.correct))
    return pairs


def _brier_and_buckets(pairs: list[tuple[float, bool]]) -> tuple[float, tuple[CalibrationBucket, ...]]:
    if not pairs:
        empty = tuple(
            CalibrationBucket(
                range_label=_bucket_label(_BUCKET_EDGES[i], _BUCKET_EDGES[i + 1]),
                count=0,
                accuracy=0.0,
            )
            for i in range(len(_BUCKET_EDGES) - 1)
        )
        return 0.0, empty

    brier = sum((conf - float(correct)) ** 2 for conf, correct in pairs) / len(pairs)
    buckets: list[CalibrationBucket] = []
    for i in range(len(_BUCKET_EDGES) - 1):
        low, high = _BUCKET_EDGES[i], _BUCKET_EDGES[i + 1]
        if high >= 1.0:
            in_bucket = [(c, ok) for c, ok in pairs if low <= c <= high]
        else:
            in_bucket = [(c, ok) for c, ok in pairs if low <= c < high]
        accuracy = (
            sum(ok for _, ok in in_bucket) / len(in_bucket) if in_bucket else 0.0
        )
        buckets.append(
            CalibrationBucket(
                range_label=_bucket_label(low, high),
                count=len(in_bucket),
                accuracy=accuracy,
            )
        )
    return brier, tuple(buckets)


class CalibrationComparison(BaseModel):
    """Side-by-side calibration for primary vs alternate confidence signals."""

    model_config = ConfigDict(frozen=True)

    student_id: str
    primary: CalibrationReport
    self_reported: CalibrationReport | None
    logprob: CalibrationReport | None


def calibration_report(state: KnowledgeState, student_id: str) -> CalibrationReport:
    """Compare the answer's primary confidence to graded correctness."""
    pairs = _collect_pairs(state, student_id, confidence_attr="confidence")
    brier, buckets = _brier_and_buckets(pairs)
    return CalibrationReport(
        student_id=student_id,
        n_samples=len(pairs),
        brier_score=brier,
        buckets=buckets,
    )


def calibration_comparison(state: KnowledgeState, student_id: str) -> CalibrationComparison:
    """Compare primary, self-reported and log-prob confidence calibrations."""
    primary = calibration_report(state, student_id)
    self_pairs = _collect_pairs(state, student_id, confidence_attr="self_reported_confidence")
    log_pairs = _collect_pairs(state, student_id, confidence_attr="logprob_confidence")
    self_brier, self_buckets = _brier_and_buckets(self_pairs)
    log_brier, log_buckets = _brier_and_buckets(log_pairs)
    return CalibrationComparison(
        student_id=student_id,
        primary=primary,
        self_reported=CalibrationReport(
            student_id=student_id,
            n_samples=len(self_pairs),
            brier_score=self_brier,
            buckets=self_buckets,
        )
        if self_pairs
        else None,
        logprob=CalibrationReport(
            student_id=student_id,
            n_samples=len(log_pairs),
            brier_score=log_brier,
            buckets=log_buckets,
        )
        if log_pairs
        else None,
    )
