"""Object identity persistence across workshop sources (M22).

Links the same object (e.g. magnet/plasma demo) appearing in workshop 3
and workshop 9 into one persistent identity registry.

    PYTHONPATH=src python3 examples/50_object_identity.py
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
    workdir = Path(tempfile.mkdtemp(prefix="allm-object-identity-"))
    store = SQLiteRecordStore(workdir / "identity.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_analytics=os.environ.get("ALLM_VISION_ANALYTICS", "1") == "1",
        enable_motion_tracking=os.environ.get("ALLM_MOTION_TRACKING", "1") == "1",
        enable_motion_continuity=os.environ.get("ALLM_MOTION_CONTINUITY", "1") == "1",
        enable_object_identity=True,
        object_identity_min_score=float(os.environ.get("ALLM_OBJECT_IDENTITY_MIN_SCORE", "0.30")),
        frames_cache_dir=workdir / "frames",
    )
    report = researcher.run_cycle()

    print("\n=== M22: Object identity persistence ===")
    print(f"  synced cues: {len(report.multimodal_synced)}")

    identities: dict[str, list] = {}
    for row in report.multimodal_synced:
        identity_id = row.object_identity_id or "unassigned"
        identities.setdefault(identity_id, []).append(row)

    for identity_id, rows in sorted(identities.items()):
        sources = sorted({row.source_id for row in rows})
        print(f"\n  identity: {identity_id}")
        print(f"    workshops: {', '.join(sources)}")
        if rows[0].identity_summary:
            print(f"    summary: {rows[0].identity_summary[:120]}...")
        for row in rows:
            print(
                f"      {row.source_id} @{row.timestamp_sec:.0f}s "
                f"linked_sources={len(row.linked_source_ids)}"
            )

    cross_source = sum(1 for row in report.multimodal_synced if row.linked_source_ids)
    print(f"\n  cross-source rows: {cross_source}/{len(report.multimodal_synced)}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.vision.identity":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
