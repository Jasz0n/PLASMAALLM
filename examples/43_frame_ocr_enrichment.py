"""Frame OCR enrichment on synced workshop evidence (M15).

    PYTHONPATH=src python3 examples/43_frame_ocr_enrichment.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.frame_ocr import tesseract_available
from allm.researcher.video_decoder import ffmpeg_available
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-frame-ocr-"))
    store = SQLiteRecordStore(workdir / "ocr.sqlite3")

    video_dir = ROOT / "transcripts/Kids/videos"
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        video_dir=video_dir if video_dir.is_dir() else None,
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_frame_ocr=True,
        enable_vision_captions=os.environ.get("ALLM_VISION_CAPTIONS", "0") == "1",
        frames_cache_dir=workdir / "frames",
        ocr_backend=os.environ.get("ALLM_OCR_BACKEND", "auto"),
    )
    report = researcher.run_cycle()

    print("\n=== M15: Frame OCR enrichment ===")
    print(f"  ffmpeg available: {ffmpeg_available()}")
    print(f"  tesseract available: {tesseract_available()}")
    print(f"  video dir: {video_dir if video_dir.is_dir() else '(none)'}")
    print(f"  synced cues: {len(report.multimodal_synced)}")

    ocr_hits = 0
    for row in report.multimodal_synced[:4]:
        visual = row.visual
        if visual is None:
            continue
        if visual.ocr_text or visual.diagram_labels:
            ocr_hits += 1
        print(f"\n  @{row.timestamp_sec:.0f}s conf={row.confidence:.2f}")
        if visual.diagram_labels:
            print(f"    labels: {', '.join(visual.diagram_labels[:6])}")
        if visual.ocr_text:
            print(f"    ocr: {visual.ocr_text[:120]}...")
        if visual.frame_path:
            print(f"    frame: {visual.frame_path}")

    print(f"\n  ocr hits: {ocr_hits}/{len(report.multimodal_synced)}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.ocr":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
