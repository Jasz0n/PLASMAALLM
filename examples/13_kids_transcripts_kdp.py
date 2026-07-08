"""M3 starter: clean and distill the 22 Kids Knowledge Seeker transcripts.

Raw files live under ``transcripts/Kids/`` (line-oriented ASR export with
timestamps). KDP Stage 2 groups speaker turns, strips broadcast noise,
and preserves character spans into the raw text.

**Training data:** use the cleaned exports — not the LLM digests::

    transcripts/Kids/cleaned/*.txt      full workshop dialogue (all speakers)
    transcripts/Kids/cleaned/mk/*.txt   Mr Keshe only, every word kept

    PYTHONPATH=src python3 examples/13_kids_transcripts_kdp.py

Optional:
    ALLM_WRITE_CLEANED=0   skip writing cleaned/*.txt
    ALLM_INJECT_GRAPH=1    inject units into SQLite + knowledge graph
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from allm.core.logging import setup_logging
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kdp.transcript_cleaning import (
    clean_transcript_document,
    render_cleaned_transcript,
    render_mk_transcript,
)
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.planner import NeedPlanner, TopicSignal
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "transcripts" / "Kids"
CLEANED_DIR = TRANSCRIPTS / "cleaned"
MK_DIR = CLEANED_DIR / "mk"


def write_cleaned_exports(documents: DocumentStore) -> tuple[int, int]:
    """Write full dialogue and MK-only full teaching exports."""
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    MK_DIR.mkdir(parents=True, exist_ok=True)
    full_count = 0
    mk_count = 0
    for doc in documents.documents():
        segments = clean_transcript_document(doc)
        full_path = CLEANED_DIR / doc.name
        full_path.write_text(render_cleaned_transcript(doc, segments) + "\n", encoding="utf-8")
        full_count += 1
        mk_text = render_mk_transcript(doc)
        if mk_text.strip():
            (MK_DIR / doc.name).write_text(mk_text + "\n", encoding="utf-8")
            mk_count += 1
    return full_count, mk_count


def main() -> None:
    setup_logging("INFO")
    if not TRANSCRIPTS.is_dir():
        raise SystemExit(f"Missing transcript directory: {TRANSCRIPTS}")

    files = sorted(TRANSCRIPTS.glob("*.txt"))
    print("\n=== Kids Knowledge Seekers — KDP ingest ===")
    print(f"  source: {TRANSCRIPTS}")
    print(f"  files:  {len(files)}")

    workdir = Path(tempfile.mkdtemp(prefix="allm-kids-kdp-"))
    record_store = SQLiteRecordStore(workdir / "kids.sqlite3")
    documents = DocumentStore(record_store)
    documents.ingest_directory(TRANSCRIPTS, context="kids-plasma")

    if os.environ.get("ALLM_WRITE_CLEANED", "1") != "0":
        full, mk = write_cleaned_exports(documents)
        print(f"  cleaned full dialogue: {full} -> {CLEANED_DIR}")
        print(f"  cleaned MK teaching:   {mk} -> {MK_DIR}")

    result = KDPipeline().distill(documents)
    print("\n=== Distillation summary ===")
    print(
        f"  {result.documents} docs -> {result.segments} segments -> "
        f"{len(result.units)} units, {len(result.conflicts)} conflict(s)"
    )
    print(f"  raw concept mentions (RCR denominator): {result.raw_units}")

    by_type: dict[str, int] = {}
    for unit in result.units:
        by_type[unit.type] = by_type.get(unit.type, 0) + 1
    print(f"  unit types: {by_type}")

    top = sorted(result.units, key=lambda u: u.confidence, reverse=True)[:8]
    print("\n  top concepts by stability:")
    for unit in top:
        print(
            f"    [{unit.type}] {unit.normalized_concept} "
            f"conf={unit.confidence:.2f} sources={len(unit.sources)}"
        )

    if result.conflicts:
        print("\n  preserved conflicts (sample):")
        for conflict in result.conflicts[:5]:
            print(f"    {conflict.concept}: {len(conflict.sources)} source(s)")

    if os.environ.get("ALLM_INJECT_GRAPH", "0") == "1":
        print("\n=== Graph injection + planner smoke test ===")
        graph = KnowledgeGraph(record_store)
        report = GraphInjector(graph, record_store).inject(result)
        print(f"  added={report['added']} revised={report['revised']} conflicts={report['conflicts']}")
        signals = [
            TopicSignal(
                topic=c.name,
                confidence=c.confidence,
                importance=c.usefulness,
                curiosity=c.curiosity,
                dependencies=c.prerequisites,
            )
            for c in graph.concepts()
        ]
        plan = NeedPlanner().plan("kids-plasma", signals)
        print("  planner top-5:")
        for item in plan.items[:5]:
            print(f"    {item.rank}. {item.topic} need={item.need:.3f}")
        kel = KnowledgeEvaluationLayer(graph, record_store, KnowledgeState(record_store)).evaluate()
        print(f"  KEL: lg={kel.lg} rcr={kel.rcr}")

    record_store.close()
    print(f"\nDone. SQLite artifacts under {workdir}")


if __name__ == "__main__":
    main()
