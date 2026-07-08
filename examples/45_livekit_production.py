"""LiveKit production integration demo (M17).

Uses SocialServer join API when available, worker buffer, and archival.

    ALLM_LIVEKIT=1 ALLM_LIVEKIT_ARCHIVE=1 PYTHONPATH=src python3 examples/45_livekit_production.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.livekit_tokens import load_livekit_config
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-livekit-prod-"))
    store = SQLiteRecordStore(workdir / "livekit.sqlite3")
    fixture = ROOT / "transcripts/Kids/visual/livekit_streams_fixture.json"

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=2,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_livekit=True,
        enable_livekit_archive=os.environ.get("ALLM_LIVEKIT_ARCHIVE", "1") == "1",
        livekit_fixture_path=fixture,
        livekit_use_worker=os.environ.get("ALLM_LIVEKIT_WORKER", "1") == "1",
        livekit_cache_dir=workdir / "livekit",
        social_api_base_url=os.environ.get("ALLM_SOCIAL_API_URL"),
        livekit_topics=(DEFAULT_TOPIC,),
    )

    config = load_livekit_config()
    if config is not None:
        creds = researcher.connect_livekit("workshop-plasma-live-demo")
        print("\n=== LiveKit credentials ===")
        print(f"  url: {creds.url}")
        print(f"  room: {creds.room_name}")
        print(f"  via: {'social join API' if os.environ.get('ALLM_SOCIAL_API_URL') else 'local token'}")

    report = researcher.run_cycle()
    worker = researcher.livekit_worker()

    print("\n=== M17: LiveKit production integration ===")
    live_rows = [row for row in report.multimodal_synced if row.is_live]
    print(f"  live evidence: {len(live_rows)}")
    print(f"  worker streams: {worker.stream_ids()}")

    archive_dir = workdir / "archives"
    if archive_dir.is_dir():
        archives = list(archive_dir.glob("*_archive.json"))
        print(f"  archived fixtures: {len(archives)}")
        for path in archives[:3]:
            print(f"    {path.name}")

    for name, yield_count, notes in report.capability_summary:
        if "livekit" in name:
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
