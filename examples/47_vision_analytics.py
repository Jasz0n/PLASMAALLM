"""Vision analytics enrichment on synced workshop evidence (M19).

Detects motion level, dominant colors, and diagram structure from fixture
metadata (stub) or OpenCV when frames are available.

    PYTHONPATH=src python3 examples/47_vision_analytics.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.vision_analytics import _opencv_available
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-vision-analytics-"))
    store = SQLiteRecordStore(workdir / "analytics.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_captions=os.environ.get("ALLM_VISION_CAPTIONS", "1") == "1",
        enable_frame_ocr=os.environ.get("ALLM_FRAME_OCR", "1") == "1",
        enable_vision_analytics=True,
        vision_analytics_backend=os.environ.get("ALLM_VISION_ANALYTICS_BACKEND", "auto"),
        frames_cache_dir=workdir / "frames",
        ocr_backend=os.environ.get("ALLM_OCR_BACKEND", "stub"),
    )
    report = researcher.run_cycle()

    print("\n=== M19: Vision analytics ===")
    print(f"  opencv available: {_opencv_available()}")
    print(f"  synced cues: {len(report.multimodal_synced)}")

    hits = 0
    for row in report.multimodal_synced[:4]:
        visual = row.visual
        if visual is None:
            continue
        if visual.visual_features or visual.analytics_summary:
            hits += 1
        print(f"\n  @{row.timestamp_sec:.0f}s conf={row.confidence:.2f}")
        if visual.motion_level:
            print(f"    motion: {visual.motion_level}")
        if visual.dominant_colors:
            print(f"    colors: {', '.join(visual.dominant_colors)}")
        print(f"    diagram: {visual.is_diagram}")
        if visual.visual_features:
            print(f"    features: {', '.join(visual.visual_features[:6])}")
        if visual.analytics_summary:
            print(f"    summary: {visual.analytics_summary[:120]}...")

    print(f"\n  analytics hits: {hits}/{len(report.multimodal_synced)}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.vision.analytics":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
