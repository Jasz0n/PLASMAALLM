"""Integration test for mediated consultation in the learning loop."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from allm.students import load_identity
from dual_consult_run import run_dual_mediated_loop


def test_dual_mediated_loop_offline() -> None:
    plasma = load_identity(ROOT / "configs/students/plasma_student.yaml")
    software = load_identity(ROOT / "configs/students/software_student.yaml")
    result = run_dual_mediated_loop(
        plasma_identity=plasma,
        software_identity=software,
        dry_run=True,
        workdir=Path(tempfile.mkdtemp(prefix="allm-test-dual-consult-")),
        verbose=False,
    )
    assert result.workdir.exists()


def test_full_multimodal_stack_offline(monkeypatch) -> None:
    """M18 — Researcher multimodal + livekit + debate + consult show-me."""
    monkeypatch.setenv("ALLM_RESEARCHER", "1")
    monkeypatch.setenv("ALLM_MULTIMODAL", "1")
    monkeypatch.setenv("ALLM_DEBATE_EVIDENCE", "1")
    monkeypatch.setenv("ALLM_CONSULT_SHOW_ME", "1")
    monkeypatch.setenv("ALLM_VISION_CAPTIONS", "1")
    monkeypatch.setenv("ALLM_AUDIO_ANALYSIS", "1")
    monkeypatch.setenv("ALLM_FRAME_OCR", "1")
    monkeypatch.setenv("ALLM_LIVEKIT", "1")
    monkeypatch.setenv("ALLM_LIVEKIT_ARCHIVE", "1")
    monkeypatch.setenv("ALLM_LIVEKIT_WORKER", "1")
    monkeypatch.setenv("ALLM_ITERATIONS", "1")

    plasma = load_identity(ROOT / "configs/students/plasma_student.yaml")
    software = load_identity(ROOT / "configs/students/software_student.yaml")
    result = run_dual_mediated_loop(
        plasma_identity=plasma,
        software_identity=software,
        dry_run=True,
        workdir=Path(tempfile.mkdtemp(prefix="allm-test-full-stack-")),
        verbose=False,
    )
    assert result.researcher_packages >= 0
    assert result.multimodal_synced >= 0
    assert result.workdir.exists()
