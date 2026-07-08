"""Combined workshop + book KEL benchmark with held-out LG (M30).

Ablation: visual delivery OFF vs ON with identical ``ALLM_LOOP_SEED`` so
exam draws match. Reports held-out gain, KEL learning gain, cross-source
alignment, and visual note delivery.

    # Offline synthetic comparison (no LLM)
    PYTHONPATH=src python3 examples/58_combined_sources_benchmark.py --dry-run

    # Full benchmark (needs Ollama / student model)
    ALLM_ITERATIONS=4 PYTHONPATH=src python3 examples/58_combined_sources_benchmark.py
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
from allm.evaluation.combined_benchmark import (
    compare_combined_benchmark_runs,
    export_combined_benchmark,
    format_combined_benchmark_report,
    metrics_from_kel_result,
    metrics_from_records,
)
from allm.evaluation import format_strategy_gain_report
from allm.loop.history import IterationHistoryWriter, LearningIterationRecord
from kids_kel_run import run_kids_kel_steered


def _apply_defaults() -> None:
    defaults = {
        "ALLM_RESEARCHER": "1",
        "ALLM_MULTIMODAL": "1",
        "ALLM_BOOK_DISCOVERY": "1",
        "ALLM_BOOK_IMAGES": "1",
        "ALLM_VISION_CAPTIONS": "1",
        "ALLM_FRAME_OCR": "1",
        "ALLM_VISION_ANALYTICS": "1",
        "ALLM_MOTION_TRACKING": "1",
        "ALLM_MOTION_CONTINUITY": "1",
        "ALLM_OBJECT_IDENTITY": "1",
        "ALLM_VISUAL_DISTILL": "1",
        "ALLM_VISUAL_EXPORT": "1",
        "ALLM_VISUAL_EXPORT_AUTO": "1",
        "ALLM_CROSS_SOURCE_VERIFY": "1",
        "ALLM_RESEARCHER_WORKSHOP_FILES": "2",
        "ALLM_BOOK_MAX_FILES": "3",
        "ALLM_BOOK_MAX_PAGES": "24",
        "ALLM_BOOK_MAX_IMAGES": "6",
        "ALLM_BOOTSTRAP": "1",
        "ALLM_ITERATIONS": "4",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _synthetic_record(iteration: int, before: float, after: float) -> LearningIterationRecord:
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
        kel_lg=0.05 if iteration == 2 else None,
    )


def dry_run_compare() -> None:
    """Offline planner sanity check using synthetic histories."""
    workdir = Path(tempfile.mkdtemp(prefix="allm-combined-benchmark-dry-"))
    print("\n=== M30 combined-source benchmark (dry-run) ===")
    print("Synthetic visual-delivery OFF vs ON comparison.\n")

    control_path = workdir / "control.jsonl"
    treatment_path = workdir / "treatment.jsonl"
    control_writer = IterationHistoryWriter(control_path)
    treatment_writer = IterationHistoryWriter(treatment_path)
    for iteration, (before, after) in enumerate([(0.10, 0.0), (0.12, 0.28)], start=1):
        control_writer.append(_synthetic_record(iteration, before, after))
        treatment_writer.append(
            _synthetic_record(iteration, before, min(1.0, after + 0.08))
        )

    control = metrics_from_records(
        "no-visual-delivery",
        visual_delivery_enabled=False,
        records=control_writer.load_all(),
        history_path=control_path,
        kel_lg=0.04,
        workshop_packages=2,
        book_packages=1,
        aligned_concepts=5,
        book_figures=6,
        student_visual_exports=2,
        visual_notes_delivered=0,
        multimodal_synced=8,
    )
    treatment = metrics_from_records(
        "visual-delivery",
        visual_delivery_enabled=True,
        records=treatment_writer.load_all(),
        history_path=treatment_path,
        kel_lg=0.11,
        workshop_packages=2,
        book_packages=3,
        aligned_concepts=6,
        book_figures=8,
        student_visual_exports=4,
        visual_notes_delivered=43,
        multimodal_synced=12,
    )
    comparison = compare_combined_benchmark_runs(
        control,
        treatment,
        variable="visual_delivery",
        loop_seed=42,
    )
    out = export_combined_benchmark(workdir / "combined_benchmark.json", comparison)
    print(format_combined_benchmark_report(comparison))
    print(f"\nWrote {out}")


def full_benchmark() -> None:
    """Run both arms with real models and export comparison."""
    base = Path(tempfile.mkdtemp(prefix="allm-combined-benchmark-"))
    loop_seed = int(os.environ.get("ALLM_LOOP_SEED", "42"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(base / "book_images")

    print("\n=== M30: control (visual delivery OFF) ===")
    os.environ["ALLM_VISUAL_DELIVERY"] = "0"
    control = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=base / "control",
        verbose=True,
    )

    print("\n=== M30: treatment (visual delivery ON) ===")
    os.environ["ALLM_VISUAL_DELIVERY"] = "1"
    treatment = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=base / "treatment",
        verbose=True,
    )

    control_metrics = metrics_from_kel_result(
        control,
        label="no-visual-delivery",
        visual_delivery_enabled=False,
    )
    treatment_metrics = metrics_from_kel_result(
        treatment,
        label="visual-delivery",
        visual_delivery_enabled=True,
    )
    comparison = compare_combined_benchmark_runs(
        control_metrics,
        treatment_metrics,
        variable="visual_delivery",
        loop_seed=loop_seed,
    )
    out = export_combined_benchmark(base / "combined_benchmark.json", comparison)

    print("\n=== Combined-source benchmark ===")
    print(format_combined_benchmark_report(comparison))
    if treatment_metrics.phase_gains:
        print("\n=== Treatment phase gains ===")
        print(format_strategy_gain_report(list(treatment_metrics.phase_gains)))
    print(f"\nComparison: {out}")
    print(f"Artifacts: {base}")


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="Combined-source KEL benchmark (M30)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="offline synthetic comparison (no LLM)",
    )
    args = parser.parse_args()
    if args.dry_run:
        dry_run_compare()
    else:
        full_benchmark()


if __name__ == "__main__":
    main()
