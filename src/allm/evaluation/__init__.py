"""Evaluation: Plan.md success metrics derived from recorded history.

Improvement, learning speed, mastery and self-correction — computed
from teacher state and episodic memory, never stored separately.
"""

from allm.evaluation.calibration import CalibrationComparison, CalibrationReport, calibration_comparison, calibration_report
from allm.evaluation.generalization import (
    HoldoutGapReport,
    diagnose_holdout_gap,
    format_holdout_gap_report,
)
from allm.evaluation.metrics import (
    StudentEvaluation,
    evaluate_student,
    improvement_per_topic,
    learning_speed,
    mastery,
    self_correction_rate,
)
from allm.evaluation.ablation import (
    AblationArmSummary,
    AblationComparison,
    compare_ablation_runs,
    export_ablation_comparison,
    format_ablation_report,
    load_arm_from_history,
    summarize_arm,
)
from allm.evaluation.combined_benchmark import (
    CombinedSourcesBenchmarkComparison,
    CombinedSourcesRunMetrics,
    compare_combined_benchmark_runs,
    export_combined_benchmark,
    format_combined_benchmark_report,
    metrics_from_kel_result,
    metrics_from_records,
)
from allm.evaluation.strategy_gain import (
    StrategyPhaseGain,
    compute_marginal_strategy_gains,
    export_strategy_phase_gains,
    format_strategy_gain_report,
)

__all__ = [
    "CalibrationComparison",
    "CalibrationReport",
    "HoldoutGapReport",
    "calibration_comparison",
    "calibration_report",
    "diagnose_holdout_gap",
    "format_holdout_gap_report",
    "StudentEvaluation",
    "AblationArmSummary",
    "AblationComparison",
    "compare_ablation_runs",
    "export_ablation_comparison",
    "format_ablation_report",
    "load_arm_from_history",
    "summarize_arm",
    "CombinedSourcesBenchmarkComparison",
    "CombinedSourcesRunMetrics",
    "compare_combined_benchmark_runs",
    "export_combined_benchmark",
    "format_combined_benchmark_report",
    "metrics_from_kel_result",
    "metrics_from_records",
    "StrategyPhaseGain",
    "compute_marginal_strategy_gains",
    "export_strategy_phase_gains",
    "format_strategy_gain_report",
    "evaluate_student",
    "improvement_per_topic",
    "learning_speed",
    "mastery",
    "self_correction_rate",
]
