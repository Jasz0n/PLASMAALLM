"""Maintenance curriculum: new + review + difficult sample mix (M39)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from allm.collector.pool import SamplePool
from allm.data.base import Sample
from allm.loop.learning_loop import LoopConfig
from allm.students.failures import FailureLog
from allm.teacher.state import KnowledgeState


@dataclass(frozen=True)
class MaintenanceSplit:
    """Fraction of each iteration budget allocated to learning modes."""

    new_fraction: float = 0.7
    review_fraction: float = 0.2
    difficult_fraction: float = 0.1

    def bucket_sizes(self, total: int) -> tuple[int, int, int]:
        """Return (new, review, difficult) sample counts."""
        if total <= 0:
            return 0, 0, 0
        review = max(0, int(round(total * self.review_fraction)))
        difficult = max(0, int(round(total * self.difficult_fraction)))
        new = max(0, total - review - difficult)
        if new == 0 and total > 0:
            new = 1
            if review > 0:
                review -= 1
            elif difficult > 0:
                difficult -= 1
        return new, review, difficult


def maintenance_split_from_env(global_ks: float | None = None) -> MaintenanceSplit | None:
    """Parse maintenance curriculum env flags, optionally adapting to KS."""
    if os.environ.get("ALLM_MAINTENANCE_CURRICULUM", "1") != "1":
        return None
    if os.environ.get("ALLM_ADAPTIVE_MAINTENANCE", "1") == "1" and global_ks is not None:
        return adaptive_maintenance_split(global_ks)
    return MaintenanceSplit(
        new_fraction=float(os.environ.get("ALLM_MAINTENANCE_NEW", "0.7")),
        review_fraction=float(os.environ.get("ALLM_MAINTENANCE_REVIEW", "0.2")),
        difficult_fraction=float(os.environ.get("ALLM_MAINTENANCE_DIFFICULT", "0.1")),
    )


def adaptive_maintenance_split(global_ks: float) -> MaintenanceSplit:
    """Shift toward review when KS is low (M40)."""
    if global_ks >= 0.85:
        return MaintenanceSplit(new_fraction=0.9, review_fraction=0.05, difficult_fraction=0.05)
    if global_ks >= 0.70:
        return MaintenanceSplit(new_fraction=0.7, review_fraction=0.2, difficult_fraction=0.1)
    if global_ks >= 0.50:
        return MaintenanceSplit(new_fraction=0.55, review_fraction=0.30, difficult_fraction=0.15)
    return MaintenanceSplit(new_fraction=0.4, review_fraction=0.40, difficult_fraction=0.20)


def mastered_topics(
    state: KnowledgeState,
    student_id: str,
    *,
    threshold: float | None = None,
) -> list[str]:
    """Topics at or above mastery threshold for review selection."""
    cutoff = threshold
    if cutoff is None:
        cutoff = float(os.environ.get("ALLM_MAINTENANCE_MASTERY", "0.25"))
    rows: list[str] = []
    for topic in state.topics(student_id):
        confidence = state.confidence(student_id, topic)
        if confidence is not None and confidence >= cutoff:
            rows.append(topic)
    return rows


def maintenance_topics_from_recommendations(recommendations: list) -> list[str]:
    """Topics flagged by Researcher for maintenance review."""
    rows: list[str] = []
    for rec in recommendations:
        kind = getattr(rec, "recommendation_kind", "discovery")
        if kind == "maintenance" and rec.topic not in rows:
            rows.append(rec.topic)
    return rows


def collect_curriculum_mix(
    *,
    pool: SamplePool,
    failures: FailureLog,
    state: KnowledgeState,
    student_id: str,
    goal_topics: list[str],
    maintenance_topics: list[str],
    cfg: LoopConfig,
    split: MaintenanceSplit,
    primary_samples: list[Sample] | None = None,
    planner_review_topics: list[str] | None = None,
) -> tuple[list[Sample], dict[str, int]]:
    """Allocate review/difficult from the full iteration budget (M41).

    Primary samples (e.g. pinned book definitions) fill only the *new*
    bucket so maintenance always receives reserved review slots.
    """
    total = cfg.samples_per_iteration
    new_count, review_count, difficult_count = split.bucket_sizes(total)
    primary = list(primary_samples or [])
    capped_primary = primary[:new_count]

    review_candidates = list(
        dict.fromkeys(
            list(planner_review_topics or [])
            + list(maintenance_topics)
            + mastered_topics(state, student_id)
        )
    )
    if not review_candidates:
        review_candidates = list(goal_topics)

    supplement_new = max(0, new_count - len(capped_primary))
    new_rows = pool.collect(
        topics=goal_topics or None,
        limit=supplement_new,
        kinds=cfg.sample_kinds or None,
    )
    if os.environ.get("ALLM_MAINTENANCE_OPTIMIZER", "1") == "1" and planner_review_topics:
        from allm.planner.maintenance_budget import collect_prioritized_review_samples

        review_rows = collect_prioritized_review_samples(
            pool,
            list(planner_review_topics),
            limit=review_count,
            kinds=cfg.sample_kinds or None,
        )
    else:
        review_rows = pool.collect(
            topics=review_candidates or None,
            limit=review_count,
            kinds=cfg.sample_kinds or None,
        )
    difficult_rows = failures.training_samples(student_id)[:difficult_count]

    collected: list[Sample] = []
    for row in (*capped_primary, *new_rows, *review_rows, *difficult_rows):
        if len(collected) >= total:
            break
        collected.append(row)

    counts = {
        "new": len(capped_primary) + len(new_rows),
        "review": len(review_rows),
        "difficult": len(difficult_rows),
        "primary": len(capped_primary),
    }
    return collected, counts


def collect_maintenance_mix(
    *,
    pool: SamplePool,
    failures: FailureLog,
    state: KnowledgeState,
    student_id: str,
    goal_topics: list[str],
    maintenance_topics: list[str],
    cfg: LoopConfig,
    split: MaintenanceSplit,
    base_samples: list[Sample],
    planner_review_topics: list[str] | None = None,
) -> tuple[list[Sample], dict[str, int]]:
    """Backward-compatible wrapper around :func:`collect_curriculum_mix`."""
    return collect_curriculum_mix(
        pool=pool,
        failures=failures,
        state=state,
        student_id=student_id,
        goal_topics=goal_topics,
        maintenance_topics=maintenance_topics,
        cfg=cfg,
        split=split,
        primary_samples=base_samples,
        planner_review_topics=planner_review_topics,
    )
