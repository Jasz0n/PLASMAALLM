"""Temporal motion tracking across frame sequences (M20).

Offline-friendly: infers motion vector and score from fixture frame spans.
OpenCV frame-diff when extracted sequences are available.

    PYTHONPATH=src python3 examples/48_motion_tracking.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.vision_analytics import _opencv_available
from allm.researcher.video_decoder import ffmpeg_available
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-motion-tracking-"))
    store = SQLiteRecordStore(workdir / "motion.sqlite3")
    video_dir = ROOT / "transcripts/Kids/videos"

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        video_dir=video_dir if video_dir.is_dir() else None,
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_analytics=os.environ.get("ALLM_VISION_ANALYTICS", "1") == "1",
        enable_motion_tracking=True,
        motion_tracking_backend=os.environ.get("ALLM_MOTION_TRACKING_BACKEND", "auto"),
        motion_tracking_samples=int(os.environ.get("ALLM_MOTION_TRACKING_SAMPLES", "3")),
        frames_cache_dir=workdir / "frames",
    )
    report = researcher.run_cycle()

    print("\n=== M20: Temporal motion tracking ===")
    print(f"  ffmpeg available: {ffmpeg_available()}")
    print(f"  opencv available: {_opencv_available()}")
    print(f"  synced cues: {len(report.multimodal_synced)}")

    hits = 0
    for row in report.multimodal_synced[:4]:
        visual = row.visual
        if visual is None:
            continue
        if visual.motion_summary:
            hits += 1
        print(f"\n  @{row.timestamp_sec:.0f}s conf={row.confidence:.2f}")
        if visual.frame_start is not None:
            print(f"    span: frames {visual.frame_start}-{visual.frame_end}")
        if visual.motion_vector:
            print(f"    vector: {visual.motion_vector}")
        if visual.motion_score is not None:
            print(f"    score: {visual.motion_score:.2f}")
        if visual.frame_sequence_paths:
            print(f"    sequence: {len(visual.frame_sequence_paths)} frames")
        if visual.motion_summary:
            print(f"    summary: {visual.motion_summary[:120]}...")

    print(f"\n  motion hits: {hits}/{len(report.multimodal_synced)}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.vision.motion":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
