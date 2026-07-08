"""Retention gates for KEL strategy advancement (M38)."""

from __future__ import annotations

from dataclasses import dataclass, field

from allm.loop.learning_loop import IterationReport
from allm.loop.strategy import LearningStrategy, profile_for
from allm.trainer.forgetting import ForgettingReport


@dataclass
class RetentionContext:
    """Signals that must be stable before KEL advances strategy."""

    heldout_peak: float = 0.0
    last_heldout: float = 0.0
    phase_peaks: dict[str, float] = field(default_factory=dict)
    current_phase: str | None = None
    forgetting_reports: tuple[ForgettingReport, ...] = ()
    retention_stable: bool = True
    block_reason: str = ""
    ks: float | None = None
    ks_stable: bool = True


class HeldoutRetentionTracker:
    """Track held-out exam peaks globally and per learning phase."""

    def __init__(self) -> None:
        self._global_peak = 0.0
        self._phase_peaks: dict[str, float] = {}

    @property
    def global_peak(self) -> float:
        return self._global_peak

    @property
    def phase_peaks(self) -> dict[str, float]:
        return dict(self._phase_peaks)

    def record(self, score: float, *, phase: str | None) -> None:
        """Update peaks after one iteration."""
        self._global_peak = max(self._global_peak, score)
        if phase is not None:
            self._phase_peaks[phase] = max(self._phase_peaks.get(phase, 0.0), score)

    def phase_peak(self, phase: str) -> float:
        return self._phase_peaks.get(phase, 0.0)


def _recent_forgetting(reports: list[IterationReport]) -> tuple[ForgettingReport, ...]:
    rows: list[ForgettingReport] = []
    for report in reports[-2:]:
        rows.extend(report.forgetting)
    return tuple(rows)


def build_retention_context(
    reports: list[IterationReport],
    tracker: HeldoutRetentionTracker,
    *,
    current_phase: str | None,
    max_drop_from_peak: float,
    require_stable: bool,
    ks: float | None = None,
    ks_threshold: float = 0.70,
) -> RetentionContext:
    """Assemble retention signals from loop history."""
    last_heldout = 0.0
    if reports and reports[-1].students:
        last_heldout = reports[-1].students[0].score_after

    peak = tracker.global_peak
    forgetting = _recent_forgetting(reports)
    stable = True
    reason = ""

    if require_stable and reports:
        drop = peak - last_heldout
        if peak > 0 and drop > max_drop_from_peak:
            stable = False
            reason = (
                f"held-out dropped {drop:.2f} from peak {peak:.2f} "
                f"(max drop {max_drop_from_peak:.2f})"
            )

    for report in forgetting:
        if report.regressions:
            stable = False
            topics = ", ".join(sorted(report.regressions)[:3])
            reason = reason or f"forgetting on topic(s): {topics}"
            break

    if current_phase == "workshop" and tracker.phase_peak("book") > 0:
        book_peak = tracker.phase_peak("book")
        if last_heldout < book_peak - max_drop_from_peak:
            stable = False
            reason = reason or (
                f"workshop phase score {last_heldout:.2f} below book peak {book_peak:.2f}"
            )

    ks_stable = True
    if ks is not None and ks < ks_threshold:
        stable = False
        ks_stable = False
        reason = reason or f"KS {ks:.2f} below {ks_threshold:.2f}"

    return RetentionContext(
        heldout_peak=peak,
        last_heldout=last_heldout,
        phase_peaks=tracker.phase_peaks,
        current_phase=current_phase,
        forgetting_reports=forgetting,
        retention_stable=stable,
        block_reason=reason,
        ks=ks,
        ks_stable=ks_stable,
    )


def reset_strategy_for_new_phase(active_config) -> object:
    """Restart curriculum at definitions when a new source phase begins."""
    profile = profile_for(LearningStrategy.DEFINITIONS)
    return active_config.model_copy(
        update={
            "strategy": LearningStrategy.DEFINITIONS.value,
            "sample_kinds": profile.sample_kinds,
            "use_exam_paraphrase": profile.use_exam_paraphrase,
            "study_failures": profile.study_failures,
        }
    )
