"""Visual knowledge package distillation for Teacher handoff (M23).

Turns enriched multimodal evidence into distilled briefs: images,
diagrams, explanations, experiments, and questions — for Teacher review,
not raw video playback for students.

    PYTHONPATH=src python3 examples/51_visual_distillation.py
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
    workdir = Path(tempfile.mkdtemp(prefix="allm-visual-distill-"))
    store = SQLiteRecordStore(workdir / "distill.sqlite3")

    researcher = ResearcherLayer(
        store,
        workshop_dir=ROOT / "transcripts/Kids/cleaned/mk",
        video_fixture_dir=ROOT / "transcripts/Kids/visual",
        workshop_max_files=5,
        catalog_topics=(DEFAULT_TOPIC,),
        enable_vision_analytics=os.environ.get("ALLM_VISION_ANALYTICS", "1") == "1",
        enable_motion_tracking=os.environ.get("ALLM_MOTION_TRACKING", "1") == "1",
        enable_motion_continuity=os.environ.get("ALLM_MOTION_CONTINUITY", "1") == "1",
        enable_object_identity=os.environ.get("ALLM_OBJECT_IDENTITY", "1") == "1",
        enable_visual_distillation=True,
        visual_distillation_max_images=int(os.environ.get("ALLM_VISUAL_DISTILL_IMAGES", "3")),
        visual_distillation_max_questions=int(os.environ.get("ALLM_VISUAL_DISTILL_QUESTIONS", "5")),
        frames_cache_dir=workdir / "frames",
    )
    report = researcher.run_cycle()

    print("\n=== M23: Visual distillation for Teacher handoff ===")
    print(f"  packages: {len(report.packages)}")
    print(f"  synced cues: {len(report.multimodal_synced)}")

    brief_count = 0
    for package in report.packages:
        briefs = package.distilled_visual_briefs
        if not briefs:
            continue
        brief_count += len(briefs)
        print(f"\n  package: {package.id}")
        print(f"    provider: {package.provider}")
        print(f"    briefs: {len(briefs)}")
        for brief in briefs[:2]:
            print(f"\n    brief: {brief.brief_id}")
            print(f"      concept: {brief.concept_name}")
            print(f"      confidence: {brief.evidence_confidence:.2f}")
            print(f"      images: {len(brief.images)}")
            for image in brief.images[:2]:
                print(f"        - {image[:100]}...")
            if brief.diagram_summary:
                print(f"      diagram: {brief.diagram_summary[:100]}...")
            if brief.explanations:
                print(f"      explanations: {len(brief.explanations)}")
            if brief.experiment_prompt:
                print(f"      experiment: {brief.experiment_prompt[:100]}...")
            print(f"      questions: {len(brief.questions)}")
            for question in brief.questions[:3]:
                print(f"        ? {question}")

    print(f"\n  total distilled briefs: {brief_count}")
    for name, yield_count, notes in report.capability_summary:
        if name == "understanding.visual.distill":
            print(f"  capability: {name} yield={yield_count} ({notes})")

    distilled_recs = [rec for rec in report.recommendations if "visual brief" in rec.reason]
    if distilled_recs:
        print(f"\n  recommendations mentioning visual briefs: {len(distilled_recs)}")

    store.close()
    print(f"\nDone. Artifacts under {workdir}")


if __name__ == "__main__":
    main()
