"""M3: Kids cleaned corpus -> knowledge graph -> training samples.

Uses the **full cleaned MK transcripts** (not digests, not raw ASR) as
the knowledge source:

    transcripts/Kids/cleaned/mk/*.txt  ->  KDP  ->  graph  ->  samples.jsonl

Reports KEL metrics (RCR, GHS) and planner rankings without raw text.

    PYTHONPATH=src python3 examples/15_kids_corpus_graph.py

Optional:
    ALLM_CORPUS=full       use cleaned/*.txt (all speakers) for KDP only
    ALLM_SAMPLES=mk|ku|both|exam  sample source (default mk; exam = definition+we_call+compact)
    ALLM_SAMPLE_KIND     filter kinds within mk (default all; or definition,we_call,compact,teaching)
    ALLM_KDP_EMBEDDINGS=1  enable pinned embedding clustering in Stage 5
    ALLM_INJECT_GRAPH=0    skip graph write (distill + export only)
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from allm.collector import SamplePool
from allm.core.logging import setup_logging
from allm.kdp import DocumentStore, GraphInjector, KDPipeline
from allm.kdp.concept_quality import concept_quality_report
from allm.kdp.corpus import DEFAULT_TOPIC, export_samples_jsonl, ingest_cleaned_corpus, load_samples_jsonl
from allm.kdp.mk_samples import dedupe_samples, mk_corpus_to_samples, parse_sample_kinds
from allm.kel import KnowledgeEvaluationLayer
from allm.knowledge import KnowledgeGraph
from allm.planner import NeedPlanner, TopicSignal
from allm.storage import SQLiteRecordStore
from allm.teacher import KnowledgeState

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT / "transcripts" / "Kids"
CLEANED_MK = TRANSCRIPTS / "cleaned" / "mk"
CLEANED_FULL = TRANSCRIPTS / "cleaned"
SAMPLES_OUT = TRANSCRIPTS / "samples.jsonl"
SAMPLES_EXAM = TRANSCRIPTS / "samples_exam.jsonl"
SAMPLES_DEFINITIONS = TRANSCRIPTS / "samples_definitions.jsonl"


def corpus_dir() -> Path:
    if os.environ.get("ALLM_CORPUS", "mk").lower() == "full":
        return CLEANED_FULL
    return CLEANED_MK


def main() -> None:
    setup_logging("INFO")
    source = corpus_dir()
    if not source.is_dir():
        raise SystemExit(
            f"Missing {source}. Run: PYTHONPATH=src python3 examples/13_kids_transcripts_kdp.py"
        )

    files = sorted(source.glob("*.txt"))
    print("\n=== M3 Kids corpus -> graph -> samples ===")
    print(f"  source: {source} ({len(files)} files)")
    print(f"  KDP embeddings: {os.environ.get('ALLM_KDP_EMBEDDINGS', '0') == '1'}")

    workdir = Path(tempfile.mkdtemp(prefix="allm-kids-graph-"))
    store = SQLiteRecordStore(workdir / "kids.sqlite3")
    documents = DocumentStore(store)
    t0 = time.perf_counter()
    ingest_cleaned_corpus(documents, source)

    result = KDPipeline().distill(documents)
    elapsed = time.perf_counter() - t0
    print(f"\n=== KDP ({elapsed:.1f}s) ===")
    print(
        f"  {result.documents} docs -> {result.segments} segments -> "
        f"{len(result.units)} units ({result.raw_units} raw mentions)"
    )
    print(f"  conflicts: {len(result.conflicts)}")

    by_type: dict[str, int] = {}
    for unit in result.units:
        by_type[unit.type] = by_type.get(unit.type, 0) + 1
    print(f"  types: {by_type}")

    concept_names = sorted({u.normalized_concept for u in result.units if u.type == "concept"})
    quality = concept_quality_report(concept_names)
    print(
        f"  concept quality: {quality['clean']}/{quality['total']} clean "
        f"({quality['clean_ratio']:.0%}), {quality['noisy']} noisy labels"
    )

    if os.environ.get("ALLM_INJECT_GRAPH", "1") != "0":
        print("\n=== Graph injection ===")
        graph = KnowledgeGraph(store)
        report = GraphInjector(graph, store).inject(result)
        print(f"  added={report['added']} revised={report['revised']} conflicts={report['conflicts']}")
        print(f"  concepts in graph: {len(graph.names())}")

        if os.environ.get("ALLM_ROUTE_STUDENTS", "1") == "1":
            from allm.planner import IngestRouter
            from allm.students import load_identities_dir

            students_dir = ROOT / "configs/students"
            if students_dir.is_dir():
                identities = load_identities_dir(students_dir)
                if identities:
                    router = IngestRouter(identities.values(), seed=42)
                    routed = router.route_document(graph.names())
                    print("\n=== Specialist routing (top concepts) ===")
                    for concept in sorted(routed)[:12]:
                        print(f"  {concept:<28} -> {', '.join(routed[concept])}")

        print("\n=== Planner (no raw text) ===")
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
        plan = NeedPlanner().plan(DEFAULT_TOPIC, signals)
        print("  top-10 topics by need:")
        for item in plan.items[:10]:
            print(f"    {item.rank:2}. {item.topic[:48]:<48} need={item.need:.3f}")

        kel = KnowledgeEvaluationLayer(graph, store, KnowledgeState(store)).evaluate(result)
        rcr = f"{kel.rcr:.2f}" if kel.rcr is not None else "—"
        ghs = f"{kel.ghs:.2f}" if kel.ghs is not None else "—"
        print(f"\n=== KEL ===")
        print(f"  RCR={rcr}  CD={kel.cd}  GHS={ghs}  units={len(result.units)}")

    samples_mode = os.environ.get("ALLM_SAMPLES", "mk").lower()
    kind_filter = parse_sample_kinds(os.environ.get("ALLM_SAMPLE_KIND"))
    if samples_mode == "exam":
        kind_filter = frozenset({"definition", "we_call", "compact"})
    if samples_mode in ("mk", "both", "exam"):
        mk_samples = mk_corpus_to_samples(CLEANED_MK, kinds=kind_filter)
        print(f"\n=== MK prose samples ===")
        print(f"  extracted: {len(mk_samples)} from {CLEANED_MK} kinds={sorted(kind_filter)}")
        if mk_samples:
            print(f"  example: {mk_samples[0].input[:70]}...")
    else:
        mk_samples = []

    ku_samples = []
    if samples_mode in ("ku", "both"):
        from allm.kdp.corpus import units_to_samples

        ku_samples = units_to_samples(list(result.units))
        print(f"\n=== KU-derived samples (legacy) ===")
        print(f"  extracted: {len(ku_samples)}")

    if samples_mode == "both":
        samples = dedupe_samples(mk_samples + ku_samples)
    elif samples_mode == "ku":
        samples = ku_samples
    elif samples_mode == "exam":
        samples = mk_samples
    else:
        samples = mk_samples

    if not samples:
        raise SystemExit("No training samples produced — check cleaned/mk/ exports")

    count = export_samples_jsonl(samples, SAMPLES_OUT)
    exam_samples = mk_corpus_to_samples(CLEANED_MK, kinds=frozenset({"definition", "we_call", "compact"}))
    def_samples = mk_corpus_to_samples(CLEANED_MK, kinds=frozenset({"definition", "we_call"}))
    exam_count = export_samples_jsonl(exam_samples, SAMPLES_EXAM)
    def_count = export_samples_jsonl(def_samples, SAMPLES_DEFINITIONS)
    pool = SamplePool()
    pool.ingest(samples)
    print(f"\n=== Training samples ({samples_mode}) ===")
    print(f"  exported: {count} labelled samples -> {SAMPLES_OUT}")
    print(f"  exam pool: {exam_count} definition/compact samples -> {SAMPLES_EXAM}")
    print(f"  definitions: {def_count} definition/we_call samples -> {SAMPLES_DEFINITIONS}")
    print(f"  pool topics: {pool.topics()[:8]}{'...' if len(pool.topics()) > 8 else ''}")

    store.close()
    print(f"\nDone. SQLite under {workdir}")


if __name__ == "__main__":
    main()
