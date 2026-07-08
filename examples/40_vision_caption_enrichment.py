"""Vision caption enrichment on synced workshop evidence (M12).

    PYTHONPATH=src python3 examples/40_vision_caption_enrichment.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.video_decoder import ffmpeg_available
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-vision-caption-"))
    store = SQLiteRecordStore(workdir / "vision.sqlite3")

    video_dir = ROOT / "transcripts/Kids/videos"
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        video_dir=video_dir if video_dir.is_dir() else None,
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_captions=True,
        frames_cache_dir=workdir / "frames",
        vision_caption_backend=os.environ.get("ALLM_VISION_BACKEND", "stub"),
    )
    report = researcher.run_cycle()

    print("\n=== M12: Vision caption enrichment ===")
    print(f"  ffmpeg available: {ffmpeg_available()}")
    print(f"  video dir: {video_dir if video_dir.is_dir() else '(none)'}")
    print(f"  synced cues: {len(report.multimodal_synced)}")

    captioned = 0
    for row in report.multimodal_synced[:4]:
        visual = row.visual
        if visual is None:
            continue
        caption_text = visual.caption or ""
        if caption_text:
            captioned += 1
        print(f"\n  @{row.timestamp_sec:.0f}s conf={row.confidence:.2f}")
        if caption_text:
            print(f"    caption: {caption_text[:100]}...")
        if visual.frame_path:
            print(f"    frame: {visual.frame_path}")

    print(f"\n  captioned: {captioned}/{len(report.multimodal_synced)}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.vision":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
