"""Kids plasma: identity ablation — mission on vs off (one variable).

Runs the same KEL-steered held-out loop twice with identical ``ALLM_LOOP_SEED``
so exam draws match. Only difference: specialist identity enabled or disabled.

    # Offline: compare existing history files
    PYTHONPATH=src python3 examples/22_identity_ablation.py --dry-run

    # Full LLM ablation (slow — needs Ollama)
    ALLM_ITERATIONS=4 PYTHONPATH=src python3 examples/22_identity_ablation.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.core.logging import setup_logging
from allm.evaluation.ablation import (
    compare_ablation_runs,
    export_ablation_comparison,
    format_ablation_report,
    load_arm_from_history,
    summarize_arm,
)
from allm.evaluation import format_strategy_gain_report
from allm.loop.history import IterationHistoryWriter
from kids_kel_run import run_kids_kel_steered


def dry_run_compare() -> None:
    """Offline planner sanity check using synthetic histories if no LLM."""
    workdir = Path(tempfile.mkdtemp(prefix="allm-ablation-dry-"))
    print("\n=== Identity ablation (dry-run mode) ===")
    print("Runs mission-off then mission-on with ALLM_ITERATIONS=0 equivalent")
    print("Use full mode for LLM held-out scores.\n")

    from allm.evaluation.strategy_gain import StrategyPhaseGain
    from allm.loop.history import LearningIterationRecord

    def _record(iteration: int, before: float, after: float) -> LearningIterationRecord:
        return LearningIterationRecord(
            iteration=iteration,
            strategy="definitions",
            sample_kinds=("definition",),
            sample_ids=("s1",),
            student_id="kids-kel",
            score_before=before,
            score_after=after,
            goals=("kids-plasma",),
            samples_studied=32,
            questions_per_exam=8,
            use_exam_paraphrase=False,
            holdout_answers_in_train=0,
        )

    control_path = workdir / "control.jsonl"
    treatment_path = workdir / "treatment.jsonl"
    control_writer = IterationHistoryWriter(control_path)
    treatment_writer = IterationHistoryWriter(treatment_path)
    for iteration, (before, after) in enumerate([(0.11, 0.0), (0.12, 0.35)], start=1):
        control_writer.append(_record(iteration, before, after))
        treatment_writer.append(_record(iteration, before, min(1.0, after + 0.05)))

    control = load_arm_from_history("no-mission", control_path, mission_enabled=False)
    treatment = load_arm_from_history("mission", treatment_path, mission_enabled=True)
    comparison = compare_ablation_runs(
        control,
        treatment,
        variable="student_identity",
        loop_seed=42,
    )
    out = export_ablation_comparison(workdir / "ablation_comparison.json", comparison)
    print(format_ablation_report(comparison))
    print(f"\nWrote {out}")


def full_ablation() -> None:
    """Run both arms with real models and export comparison."""
    base = Path(tempfile.mkdtemp(prefix="allm-identity-ablation-"))
    loop_seed = int(os.environ.get("ALLM_LOOP_SEED", "42"))

    print("\n=== Identity ablation: control (no mission) ===")
    control = run_kids_kel_steered(
        identity_path="0",
        workdir=base / "control",
        verbose=True,
    )

    print("\n=== Identity ablation: treatment (plasma mission) ===")
    treatment = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=base / "treatment",
        verbose=True,
    )

    control_arm = summarize_arm(
        "no-mission",
        mission_enabled=False,
        records=IterationHistoryWriter(control.history_path).load_all(),
        history_path=control.history_path,
        kel_lg=control.kel_lg,
    )
    treatment_arm = summarize_arm(
        "mission",
        mission_enabled=True,
        records=IterationHistoryWriter(treatment.history_path).load_all(),
        history_path=treatment.history_path,
        kel_lg=treatment.kel_lg,
    )
    comparison = compare_ablation_runs(
        control_arm,
        treatment_arm,
        variable="student_identity",
        loop_seed=loop_seed,
    )
    out = export_ablation_comparison(base / "ablation_comparison.json", comparison)

    print("\n=== Ablation comparison ===")
    print(format_ablation_report(comparison))
    print("\n=== Treatment phase gains ===")
    print(format_strategy_gain_report(list(treatment_arm.phase_gains)))
    print(f"\nComparison: {out}")
    print(f"Artifacts: {base}")


def main() -> None:
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Identity on vs off ablation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="offline synthetic comparison (no LLM)",
    )
    args = parser.parse_args()
    if args.dry_run:
        dry_run_compare()
    else:
        full_ablation()


if __name__ == "__main__":
    main()
