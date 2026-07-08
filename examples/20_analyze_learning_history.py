"""Analyze learning_history.jsonl — marginal gain per strategy phase.

    PYTHONPATH=src python3 examples/20_analyze_learning_history.py /path/to/learning_history.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

from allm.evaluation import (
    compute_marginal_strategy_gains,
    export_strategy_phase_gains,
    format_strategy_gain_report,
)
from allm.loop.history import IterationHistoryWriter


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 examples/20_analyze_learning_history.py <history.jsonl>")

    history_path = Path(sys.argv[1])
    records = IterationHistoryWriter(history_path).load_all()
    if not records:
        raise SystemExit(f"No records in {history_path}")

    phases = compute_marginal_strategy_gains(records)
    gains_path = history_path.with_name("strategy_phase_gains.json")
    export_strategy_phase_gains(gains_path, records)

    print(f"Records: {len(records)}  Phases: {len(phases)}")
    print(f"Wrote {gains_path}\n")
    print("=== Marginal learning gain by strategy ===")
    print(format_strategy_gain_report(phases))


if __name__ == "__main__":
    main()
