"""Ollama vision caption demo — uses llava when daemon + frames exist (M13).

    # Stub fallback (no Ollama required)
    PYTHONPATH=src python3 examples/41_ollama_vision_caption.py

    # Real vision model (requires Ollama + llava + workshop MP4 frames)
    ALLM_VISION_BACKEND=auto ALLM_VISION_MODEL=llava \\
      ALLM_VISION_CAPTIONS=1 PYTHONPATH=src python3 examples/41_ollama_vision_caption.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.ollama_vision import ollama_reachable
from allm.researcher.video_decoder import ffmpeg_available
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-ollama-vision-"))
    store = SQLiteRecordStore(workdir / "ollama-vision.sqlite3")

    backend = os.environ.get("ALLM_VISION_BACKEND", "auto")
    model = os.environ.get("ALLM_VISION_MODEL", "llava")
    video_dir = ROOT / "transcripts/Kids/videos"

    print("\n=== M13: Ollama vision captions ===")
    print(f"  backend: {backend}")
    print(f"  model: {model}")
    print(f"  ollama reachable: {ollama_reachable()}")
    print(f"  ffmpeg available: {ffmpeg_available()}")
    print(f"  video dir: {video_dir if video_dir.is_dir() else '(none — transcript-only captions)'}")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        video_dir=video_dir if video_dir.is_dir() else None,
        workshop_max_files=2,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_captions=True,
        frames_cache_dir=workdir / "frames",
        vision_caption_backend=backend,
        vision_ollama_model=model,
    )
    report = researcher.run_cycle()

    vision_captions = 0
    ollama_captions = 0
    for row in report.multimodal_synced[:5]:
        visual = row.visual
        if visual is None or not visual.caption:
            continue
        vision_captions += 1
        if visual.caption.startswith("Vision:"):
            ollama_captions += 1
        print(f"\n  @{row.timestamp_sec:.0f}s conf={row.confidence:.2f}")
        print(f"    {visual.caption[:120]}...")
        if visual.frame_path:
            print(f"    frame: {visual.frame_path}")

    print(f"\n  captioned: {vision_captions}  ollama: {ollama_captions}")
    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
