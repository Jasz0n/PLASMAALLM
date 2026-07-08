"""Auto-generate video timeline fixtures from workshop transcripts (M10).

    PYTHONPATH=src python3 examples/37_auto_video_fixtures.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.video_decoder import (
    ensure_workshop_fixtures,
    ffmpeg_available,
    find_video_mentions,
    generate_fixture_from_transcript,
)
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]
WORKSHOP_DIR = ROOT / "transcripts/Kids/cleaned/mk"
TRANSCRIPT = WORKSHOP_DIR / "knowledgeSeekerWorkshop9.txt"


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-auto-video-"))
    cache_dir = workdir / "visual_cache"
    store = SQLiteRecordStore(workdir / "auto.sqlite3")

    text = TRANSCRIPT.read_text(encoding="utf-8")
    mentions = find_video_mentions(text)
    print("\n=== M10: Auto video fixture generation ===")
    print(f"  ffmpeg available: {ffmpeg_available()}")
    print(f"  video mentions in workshop9: {len(mentions)}")

    fixture = generate_fixture_from_transcript(TRANSCRIPT, curriculum_topic=DEFAULT_TOPIC)
    assert fixture is not None
    print(f"  generated cues: {len(fixture.cues)}")
    for cue in fixture.cues[:3]:
        print(f"    @{cue.timestamp_sec:.0f}s: {cue.transcript_phrase!r}")

    generated = ensure_workshop_fixtures(WORKSHOP_DIR, cache_dir, curriculum_topic=DEFAULT_TOPIC)
    print(f"\n  auto fixtures written: {len(generated)}")
    print(f"  cache dir: {cache_dir}")

    researcher = ResearcherLayer(
        store,
        workshop_dir=WORKSHOP_DIR,
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
        video_fixture_dir=cache_dir,
        auto_generate_video_fixtures=True,
    )
    report = researcher.run_cycle()
    print(f"\n  researcher synced: {len(report.multimodal_synced)}")
    if report.packages:
        pkg = report.packages[0]
        print(f"  package multimodal cues: {len(pkg.multimodal_evidence)}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
