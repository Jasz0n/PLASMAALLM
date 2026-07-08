"""Cross-cue motion continuity across workshop timeline (M21).

Links visual evidence rows that track the same object or motion thread,
e.g. magnet rotation @712s → magnet chase @845s → repulsion @848s.

    PYTHONPATH=src python3 examples/49_motion_continuity.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-motion-continuity-"))
    store = SQLiteRecordStore(workdir / "continuity.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=3,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_analytics=os.environ.get("ALLM_VISION_ANALYTICS", "1") == "1",
        enable_motion_tracking=os.environ.get("ALLM_MOTION_TRACKING", "1") == "1",
        enable_motion_continuity=True,
        motion_continuity_min_score=float(
            os.environ.get("ALLM_MOTION_CONTINUITY_MIN_SCORE", "0.35")
        ),
        frames_cache_dir=workdir / "frames",
    )
    report = researcher.run_cycle()

    print("\n=== M21: Cross-cue motion continuity ===")
    print(f"  synced cues: {len(report.multimodal_synced)}")

    tracks: dict[str, list] = {}
    for row in report.multimodal_synced:
        track_id = row.motion_track_id or "untracked"
        tracks.setdefault(track_id, []).append(row)

    for track_id, rows in sorted(tracks.items()):
        timestamps = [row.timestamp_sec for row in rows]
        print(f"\n  track: {track_id}")
        print(f"    cues: {len(rows)} @ {', '.join(f'{ts:.0f}s' for ts in timestamps)}")
        if rows[0].continuity_summary:
            print(f"    summary: {rows[0].continuity_summary[:120]}...")
        for row in rows:
            visual = row.visual
            vector = visual.motion_vector if visual else None
            print(f"      @{row.timestamp_sec:.0f}s vector={vector} linked={len(row.linked_cue_timestamps)}")

    linked_rows = sum(1 for row in report.multimodal_synced if row.linked_cue_timestamps)
    print(f"\n  linked rows: {linked_rows}/{len(report.multimodal_synced)}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.vision.continuity":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
