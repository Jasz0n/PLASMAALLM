"""Audio feature enrichment on synced workshop evidence (M14).

    PYTHONPATH=src python3 examples/42_audio_enrichment.py
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
    workdir = Path(tempfile.mkdtemp(prefix="allm-audio-enrich-"))
    store = SQLiteRecordStore(workdir / "audio.sqlite3")

    video_dir = ROOT / "transcripts/Kids/videos"
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        video_dir=video_dir if video_dir.is_dir() else None,
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_audio_analysis=True,
        audio_cache_dir=workdir / "audio",
        audio_analysis_backend=os.environ.get("ALLM_AUDIO_BACKEND", "auto"),
    )
    report = researcher.run_cycle()

    print("\n=== M14: Audio feature enrichment ===")
    print(f"  ffmpeg available: {ffmpeg_available()}")
    print(f"  video dir: {video_dir if video_dir.is_dir() else '(none)'}")
    print(f"  synced cues: {len(report.multimodal_synced)}")

    analyzed = 0
    for row in report.multimodal_synced[:4]:
        audio = row.audio
        if audio is None:
            continue
        if audio.features:
            analyzed += 1
        print(f"\n  @{row.timestamp_sec:.0f}s conf={row.confidence:.2f}")
        if audio.features:
            print(f"    features: {', '.join(audio.features)}")
        if audio.analysis:
            print(f"    analysis: {audio.analysis[:120]}...")
        if audio.clip_path:
            print(f"    clip: {audio.clip_path}")

    print(f"\n  analyzed: {analyzed}/{len(report.multimodal_synced)}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.audio":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
