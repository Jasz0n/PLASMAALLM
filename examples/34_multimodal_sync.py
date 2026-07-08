"""Multimodal sync demo — video timeline fixture + workshop transcript.

    PYTHONPATH=src python3 examples/34_multimodal_sync.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp.corpus import DEFAULT_TOPIC
from allm.researcher import ResearcherLayer
from allm.researcher.multimodal import retrieve_synced_evidence
from allm.storage import SQLiteRecordStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    setup_logging("INFO")
    workdir = Path(tempfile.mkdtemp(prefix="allm-multimodal-"))
    store = SQLiteRecordStore(workdir / "multimodal.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
    )
    report = researcher.run_cycle()

    print("\n=== Multimodal Researcher cycle (M7) ===")
    print(f"  synced evidence cues: {len(report.multimodal_synced)}")
    for row in report.multimodal_synced[:3]:
        visual = row.visual.description if row.visual else "(no visual)"
        print(f"    @{row.timestamp_sec:.0f}s conf={row.confidence:.2f}: {visual}")

    packages_with_visuals = [pkg for pkg in report.packages if pkg.multimodal_evidence]
    print(f"\n  packages with multimodal evidence: {len(packages_with_visuals)}")
    if packages_with_visuals:
        pkg = packages_with_visuals[0]
        print(f"  package {pkg.id}: {len(pkg.multimodal_evidence)} synced cues")

        print("\n=== Debate: 'Show me blue plasma' ===")
        hits = retrieve_synced_evidence(pkg, query="blue plasma", limit=2)
        for hit in hits:
            print(f"  Workshop {hit.source_id} @ {hit.timestamp_sec:.0f}s")
            if hit.visual:
                print(f"    frames {hit.visual.frame_start}-{hit.visual.frame_end}")
                print(f"    {hit.visual.description}")
            print(f"    excerpt: {hit.transcript_excerpt[:80]}...")
            print(f"    confidence: {hit.confidence:.2f}")

    print("\n=== Capability summary (multimodal) ===")
    for name, yield_count, notes in report.capability_summary:
        if "video" in name or "sync" in name:
            print(f"  {name}: yield={yield_count} ({notes})")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
