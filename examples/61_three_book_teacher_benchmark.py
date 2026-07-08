"""Three-book corpus + Teacher UI vs auto-export benchmark (M33).

Compares auto-export visual delivery against Teacher UI selective approval
on the full 3-book + workshop stack with identical ``ALLM_LOOP_SEED``.

    # Offline synthetic comparison
    PYTHONPATH=src python3 examples/61_three_book_teacher_benchmark.py --dry-run

    # Full benchmark (needs Ollama)
    ALLM_ITERATIONS=2 PYTHONPATH=src python3 examples/61_three_book_teacher_benchmark.py
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
from allm.researcher.book_corpus import audit_book_corpus, format_corpus_audit
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
        "ALLM_CROSS_SOURCE_VERIFY": "1",
        "ALLM_RESEARCHER_WORKSHOP_FILES": "2",
        "ALLM_BOOK_MAX_FILES": "3",
        "ALLM_BOOK_BOOTSTRAP_SIDECARS": "1",
        "ALLM_BOOK_MAX_PAGES": "24",
        "ALLM_BOOK_MAX_IMAGES": "6",
        "ALLM_BOOTSTRAP": "1",
        "ALLM_ITERATIONS": "2",
        "ALLM_VISUAL_DELIVERY": "1",
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
    )


def _print_corpus_audit() -> None:
    books_dir = Path(os.environ.get("ALLM_BOOK_DIR", str(ROOT / "books")))
    max_files = int(os.environ.get("ALLM_BOOK_MAX_FILES", "3"))
    print(f"\n=== Book corpus audit ({books_dir}) ===")
    entries = audit_book_corpus(books_dir, max_files=max_files)
    print(format_corpus_audit(entries))


def _configure_export_mode(mode: str) -> None:
    if mode == "teacher_ui":
        os.environ["ALLM_TEACHER_UI_APPROVAL"] = "1"
        os.environ["ALLM_VISUAL_EXPORT"] = "0"
        os.environ["ALLM_VISUAL_EXPORT_AUTO"] = "0"
    else:
        os.environ["ALLM_TEACHER_UI_APPROVAL"] = "0"
        os.environ["ALLM_VISUAL_EXPORT"] = "1"
        os.environ["ALLM_VISUAL_EXPORT_AUTO"] = "1"


def dry_run_compare() -> None:
    """Offline synthetic auto-export vs Teacher UI comparison."""
    workdir = Path(tempfile.mkdtemp(prefix="allm-three-book-benchmark-dry-"))
    print("\n=== M33 three-book benchmark (dry-run) ===")
    _print_corpus_audit()

    auto_path = workdir / "auto.jsonl"
    teacher_path = workdir / "teacher_ui.jsonl"
    auto_writer = IterationHistoryWriter(auto_path)
    teacher_writer = IterationHistoryWriter(teacher_path)
    for iteration, (before, after) in enumerate([(0.10, 0.0), (0.12, 0.30)], start=1):
        auto_writer.append(_synthetic_record(iteration, before, after))
        teacher_writer.append(_synthetic_record(iteration, before, min(1.0, after + 0.04)))

    auto = metrics_from_records(
        "auto-export",
        visual_delivery_enabled=True,
        records=auto_writer.load_all(),
        history_path=auto_path,
        kel_lg=0.10,
        workshop_packages=1,
        book_packages=1,
        aligned_concepts=6,
        book_figures=8,
        student_visual_exports=5,
        visual_notes_delivered=40,
        export_mode="auto",
        multimodal_synced=12,
    )
    teacher = metrics_from_records(
        "teacher-ui",
        visual_delivery_enabled=True,
        records=teacher_writer.load_all(),
        history_path=teacher_path,
        kel_lg=0.07,
        workshop_packages=1,
        book_packages=1,
        aligned_concepts=6,
        book_figures=8,
        student_visual_exports=2,
        visual_notes_delivered=20,
        teacher_ui_approval=True,
        teacher_approved_briefs=2,
        teacher_rejected_briefs=3,
        export_mode="teacher_ui",
        multimodal_synced=12,
    )
    comparison = compare_combined_benchmark_runs(
        auto,
        teacher,
        variable="export_mode",
        loop_seed=42,
    )
    out = export_combined_benchmark(workdir / "three_book_benchmark.json", comparison)
    print(format_combined_benchmark_report(comparison))
    print(f"\nWrote {out}")


def full_benchmark() -> None:
    """Run auto-export vs Teacher UI arms with real models."""
    base = Path(tempfile.mkdtemp(prefix="allm-three-book-benchmark-"))
    loop_seed = int(os.environ.get("ALLM_LOOP_SEED", "42"))
    os.environ["ALLM_BOOK_IMAGES_CACHE"] = str(base / "book_images")

    _print_corpus_audit()

    print("\n=== M33: control (auto-export + delivery) ===")
    _configure_export_mode("auto")
    auto = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=base / "auto",
        verbose=True,
    )

    print("\n=== M33: treatment (Teacher UI + delivery) ===")
    _configure_export_mode("teacher_ui")
    teacher = run_kids_kel_steered(
        identity_path="configs/students/kids_kel_plasma.yaml",
        workdir=base / "teacher_ui",
        verbose=True,
    )

    auto_metrics = metrics_from_kel_result(
        auto,
        label="auto-export",
        visual_delivery_enabled=True,
        export_mode="auto",
    )
    teacher_metrics = metrics_from_kel_result(
        teacher,
        label="teacher-ui",
        visual_delivery_enabled=True,
        export_mode="teacher_ui",
    )
    comparison = compare_combined_benchmark_runs(
        auto_metrics,
        teacher_metrics,
        variable="export_mode",
        loop_seed=loop_seed,
    )
    out = export_combined_benchmark(base / "three_book_benchmark.json", comparison)

    print("\n=== Three-book export-mode benchmark ===")
    print(format_combined_benchmark_report(comparison))
    if teacher_metrics.phase_gains:
        print("\n=== Teacher UI phase gains ===")
        print(format_strategy_gain_report(list(teacher_metrics.phase_gains)))
    print(f"\nComparison: {out}")
    print(f"Artifacts: {base}")


def main() -> None:
    setup_logging("INFO")
    _apply_defaults()
    parser = argparse.ArgumentParser(description="Three-book Teacher UI benchmark (M33)")
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
