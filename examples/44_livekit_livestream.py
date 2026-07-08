"""LiveKit live stream observation for the Researcher (M16).

Connects the Researcher to the same LiveKit rooms used by the app
(SocialServer streaming). Offline mode uses ``livekit_streams_fixture.json``.

    ALLM_LIVEKIT=1 PYTHONPATH=src python3 examples/44_livekit_livestream.py

With a running SocialServer and live stream:

    ALLM_LIVEKIT=1 \\
    ALLM_SOCIAL_API_URL=http://localhost:3000 \\
    LIVEKIT_URL=wss://your-app.livekit.cloud \\
    LIVEKIT_API_KEY=... \\
    LIVEKIT_API_SECRET=... \\
    PYTHONPATH=src python3 examples/44_livekit_livestream.py
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
    workdir = Path(tempfile.mkdtemp(prefix="allm-livekit-"))
    store = SQLiteRecordStore(workdir / "livekit.sqlite3")

    fixture = ROOT / "transcripts/Kids/visual/livekit_streams_fixture.json"
    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=2,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_livekit=True,
        livekit_fixture_path=fixture,
        livekit_observer_backend=os.environ.get("ALLM_LIVEKIT_BACKEND", "auto"),
        livekit_cache_dir=workdir / "livekit",
        social_api_base_url=os.environ.get("ALLM_SOCIAL_API_URL"),
        livekit_topics=(DEFAULT_TOPIC,),
        enable_vision_captions=os.environ.get("ALLM_VISION_CAPTIONS", "0") == "1",
        enable_audio_analysis=os.environ.get("ALLM_AUDIO_ANALYSIS", "0") == "1",
        enable_frame_ocr=os.environ.get("ALLM_FRAME_OCR", "0") == "1",
    )

    config = load_livekit_config()
    if config is not None:
        creds = researcher.connect_livekit("workshop-plasma-live-demo")
        print("\n=== LiveKit credentials (observer) ===")
        print(f"  url: {creds.url}")
        print(f"  room: {creds.room_name}")
        print(f"  identity: {creds.identity}")
        print(f"  token segments: {creds.token.count('.') + 1}")

    report = researcher.run_cycle()

    print("\n=== M16: LiveKit live stream observation ===")
    print(f"  livekit config: {'yes' if config else 'fixture-only'}")
    print(f"  social api: {os.environ.get('ALLM_SOCIAL_API_URL', '(none)')}")
    live_rows = [row for row in report.multimodal_synced if row.is_live]
    print(f"  live evidence rows: {len(live_rows)} / {len(report.multimodal_synced)}")

    for row in live_rows[:3]:
        print(f"\n  live @{row.timestamp_sec:.0f}s stream={row.live_stream_id} conf={row.confidence:.2f}")
        if row.visual:
            print(f"    visual: {row.visual.description[:100]}...")
        if row.audio:
            print(f"    audio: {row.audio.description[:80]}...")

    for name, yield_count, notes in report.capability_summary:
        if name in {"discovery.livekit", "understanding.livestream"}:
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
