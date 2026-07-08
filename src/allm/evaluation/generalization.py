"""Diagnose held-out generalization gaps between train and test curricula."""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from allm.data.base import Sample
from allm.exam.grading import normalise
from allm.kdp.holdout import sample_source, workshop_number

_TOKEN = re.compile(r"[a-z0-9]+")
_OVERLAP_HIGH = 0.5
_OVERLAP_LOW = 0.2


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(normalise(text)))


def prompt_overlap(left: str, right: str) -> float:
    """Token Jaccard similarity between two prompts."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _best_train_overlap(prompt: str, train_inputs: list[str]) -> float:
    if not train_inputs:
        return 0.0
    return max(prompt_overlap(prompt, train_input) for train_input in train_inputs)


class HoldoutSampleDiagnosis(BaseModel):
    """How one held-out sample relates to the train pool."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    workshop: int
    sample_kind: str
    category: str
    train_overlap: float
    answer_in_train: bool


class HoldoutGapReport(BaseModel):
    """Summary of train vs hold-out curriculum overlap."""

    model_config = ConfigDict(frozen=True)

    train_count: int
    holdout_count: int
    exact_prompt_matches: int
    high_overlap: int
    low_overlap: int
    novel_lexical: int
    answers_in_train: int
    by_category: dict[str, int] = Field(default_factory=dict)
    by_workshop: dict[int, int] = Field(default_factory=dict)
    samples: tuple[HoldoutSampleDiagnosis, ...] = ()


def _categorise(overlap: float, exact: bool) -> str:
    if exact:
        return "exact_prompt"
    if overlap >= _OVERLAP_HIGH:
        return "high_overlap"
    if overlap >= _OVERLAP_LOW:
        return "low_overlap"
    return "novel_lexical"


def diagnose_holdout_gap(train: list[Sample], holdout: list[Sample]) -> HoldoutGapReport:
    """Measure how much held-out prompts overlap train (memorization vs true generalization)."""
    train_inputs = [normalise(sample.input) for sample in train]
    train_input_set = set(train_inputs)
    train_answers = {normalise(sample.target or "") for sample in train if sample.target}

    diagnoses: list[HoldoutSampleDiagnosis] = []
    category_counts: Counter[str] = Counter()
    workshop_counts: Counter[int] = Counter()

    exact = high = low = novel = answers = 0
    for sample in holdout:
        normalized = normalise(sample.input)
        overlap = _best_train_overlap(sample.input, train_inputs)
        is_exact = normalized in train_input_set
        category = _categorise(overlap, is_exact)
        answer_present = normalise(sample.target or "") in train_answers
        workshop = workshop_number(sample_source(sample))

        if is_exact:
            exact += 1
        elif category == "high_overlap":
            high += 1
        elif category == "low_overlap":
            low += 1
        else:
            novel += 1
        if answer_present:
            answers += 1

        category_counts[category] += 1
        workshop_counts[workshop] += 1
        diagnoses.append(
            HoldoutSampleDiagnosis(
                sample_id=sample.id,
                workshop=workshop,
                sample_kind=str((sample.metadata or {}).get("sample_kind", "unknown")),
                category=category,
                train_overlap=round(overlap, 4),
                answer_in_train=answer_present,
            )
        )

    return HoldoutGapReport(
        train_count=len(train),
        holdout_count=len(holdout),
        exact_prompt_matches=exact,
        high_overlap=high,
        low_overlap=low,
        novel_lexical=novel,
        answers_in_train=answers,
        by_category=dict(category_counts),
        by_workshop=dict(sorted(workshop_counts.items())),
        samples=tuple(diagnoses),
    )


def format_holdout_gap_report(report: HoldoutGapReport) -> str:
    """Human-readable summary for loop examples."""
    novel_pct = (
        100.0 * report.novel_lexical / report.holdout_count if report.holdout_count else 0.0
    )
    lines = [
        f"  train={report.train_count} holdout={report.holdout_count}",
        f"  prompt overlap: exact={report.exact_prompt_matches} "
        f"high={report.high_overlap} low={report.low_overlap} "
        f"novel={report.novel_lexical} ({novel_pct:.0f}% novel wording)",
        f"  holdout answers also in train pool: {report.answers_in_train}/{report.holdout_count}",
    ]
    if report.by_workshop:
        workshops = ", ".join(f"w{k}:{v}" for k, v in report.by_workshop.items())
        lines.append(f"  holdout by workshop: {workshops}")
    return "\n".join(lines)
