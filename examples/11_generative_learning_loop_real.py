"""M1 benchmark: generative exams through the full learning loop.

Same as ``06_continuous_learning_loop_real.py`` but uses
:class:`ModelExamGenerator` (7B writer) instead of dataset-backed exams.

    ALLM_ITERATIONS=5 PYTHONPATH=src python3 examples/11_generative_learning_loop_real.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("ALLM_EXAM", "generative")
os.environ.setdefault("ALLM_ITERATIONS", "5")

_spec = importlib.util.spec_from_file_location(
    "loop_real",
    ROOT / "examples" / "06_continuous_learning_loop_real.py",
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

if __name__ == "__main__":
    _module.main()
