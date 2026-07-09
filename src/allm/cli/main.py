"""``allm`` command-line interface.

Built on ``argparse`` (stdlib) so the CLI has zero extra dependencies
and stays trivially replaceable. Each subcommand is a small function
``(args) -> int``; printing goes through the passed ``print`` only in
one place per command so commands stay testable.

Commands:
    allm info                       version + registered plugins
    allm config show [-c FILE]     resolved configuration as JSON
    allm plugins                    list every registry's entries
    allm runs [-c FILE]            list experiment runs
    allm model validate SPEC.yaml   validate a model spec
    allm dataset peek SPEC.yaml     print the first samples of a dataset
    allm kdp distill FILES...       distill documents into knowledge units
    allm benchmark                  state-of-the-system report (M47)
    allm audit --db FILE            the append-only write trail (M50)
    allm db backup|restore|verify   database maintenance (M50)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import allm
from allm.core.config import load_config
from allm.core.logging import setup_logging
from allm.core.registry import load_entrypoint_plugins
from allm.data.base import DatasetSpec, dataset_loaders, load_dataset
from allm.models.base import ModelSpec, model_loaders
from allm.storage.base import storage_backends
from allm.tracking.base import tracker_backends
from allm.tracking.local import LocalTracker


def _cmd_info(args: argparse.Namespace) -> int:
    print(f"allm {allm.__version__}")
    print(f"python {sys.version.split()[0]}")
    plugins = load_entrypoint_plugins()
    print(f"entry-point plugins: {', '.join(plugins) if plugins else '<none>'}")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    config = load_config(args.config).resolved()
    print(config.model_dump_json(indent=2))
    return 0


def _cmd_plugins(args: argparse.Namespace) -> int:
    load_entrypoint_plugins()
    for registry in (model_loaders, dataset_loaders, storage_backends, tracker_backends):
        print(f"{registry.kind}: {', '.join(registry.names()) or '<none>'}")
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    config = load_config(args.config).resolved()
    tracker = LocalTracker(config.tracking.root)
    runs = tracker.list_runs()
    if not runs:
        print(f"no runs under {config.tracking.root}")
    for run_id in runs:
        print(run_id)
    return 0


def _cmd_model_validate(args: argparse.Namespace) -> int:
    spec = ModelSpec.from_yaml(args.spec)
    known = spec.provider in model_loaders
    print(json.dumps(spec.model_dump(), indent=2))
    print(f"provider {spec.provider!r} registered: {known}")
    return 0 if known else 1


def _cmd_dataset_peek(args: argparse.Namespace) -> int:
    spec = DatasetSpec.from_yaml(args.spec)
    for count, sample in enumerate(load_dataset(spec), start=1):
        print(sample.model_dump_json())
        if count >= args.n:
            break
    return 0


def _cmd_kdp_distill(args: argparse.Namespace) -> int:
    from allm.kdp import DocumentStore, GraphInjector, KDPipeline
    from allm.knowledge import KnowledgeGraph
    from allm.storage import SQLiteRecordStore

    store = SQLiteRecordStore(args.db) if args.db else None
    documents = DocumentStore(store)
    for file in args.files:
        documents.ingest_file(file)
    result = KDPipeline().distill(documents)
    for unit in result.units:
        print(
            f"{unit.id}  [{unit.type}] {unit.normalized_concept}  "
            f"confidence={unit.confidence:.2f}  sources={len(unit.sources)}"
        )
    for conflict in result.conflicts:
        print(f"CONFLICT: {conflict.concept} ({len(conflict.sources)} sources disagree)")
    if store is not None:
        report = GraphInjector(KnowledgeGraph(store), store).inject(result)
        print(
            f"graph: {report['added']} added, {report['revised']} revised, "
            f"{report['conflicts']} conflict(s) stored -> {args.db}"
        )
        store.close()
    print(f"{result.documents} document(s) -> {result.segments} segment(s) "
          f"-> {len(result.units)} unit(s), {len(result.conflicts)} conflict(s)")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from pathlib import Path

    from allm.benchmarks import STANDARD_CORPORA, run_system_benchmark

    corpora = tuple(
        name.strip() for name in args.corpora.split(",") if name.strip()
    ) or STANDARD_CORPORA
    report = run_system_benchmark(
        corpora,
        iterations=args.iterations,
        seed=args.seed,
        limit=args.limit,
        student=args.student,
        root=Path(args.root) if args.root else None,
    )
    print(report.to_markdown())
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report.model_dump_json(indent=2) + "\n")
        print(f"\nreport written to {args.output}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from allm.storage import SQLiteRecordStore

    store = SQLiteRecordStore(args.db)
    try:
        records = store.audit(args.namespace, limit=args.limit, offset=args.offset)
        for record in records:
            print(
                f"{record.created_at.isoformat()}  {record.namespace}/{record.key}"
                f"  v{record.version}  {record.reason or '<no reason>'}"
            )
        if not records:
            print("no records")
    finally:
        store.close()
    return 0


def _cmd_db_backup(args: argparse.Namespace) -> int:
    from allm.storage import SQLiteRecordStore
    from allm.storage.maintenance import verify_database

    store = SQLiteRecordStore(args.db)
    try:
        destination = store.backup_to(args.destination)
    finally:
        store.close()
    ok, detail = verify_database(destination)
    print(f"backup {destination}: {detail}")
    return 0 if ok else 1


def _cmd_db_restore(args: argparse.Namespace) -> int:
    from allm.storage.maintenance import restore_database

    try:
        target = restore_database(args.backup, args.db, force=args.force)
    except (ValueError, FileExistsError) as exc:
        print(f"restore refused: {exc}")
        return 1
    print(f"restored {target}")
    return 0


def _cmd_db_verify(args: argparse.Namespace) -> int:
    from allm.storage.maintenance import verify_database

    ok, detail = verify_database(args.db)
    print(f"{args.db}: {detail}")
    return 0 if ok else 1


def _cmd_seed(args: argparse.Namespace) -> int:
    """Populate a store by running the public-loop scenario (M52)."""
    from allm.seed import seed_public_loop
    from allm.storage import SQLiteRecordStore

    store = SQLiteRecordStore(args.db)
    try:
        if store.namespaces() and not args.force:
            print(f"{args.db} already has data — pass --force to seed anyway")
            return 1
        report = seed_public_loop(store)
    finally:
        store.close()
    print(f"seeded {args.db}:")
    print(f"  concepts: {', '.join(report.concepts)}")
    print(
        f"  contested {report.contested_concept!r} -> {report.proposal_outcome} "
        f"(confidence {report.confidence_before:.2f} -> {report.confidence_after:.2f})"
    )
    print(f"  {report.events} events on the live feed")
    return 0


def _cmd_wire(args: argparse.Namespace) -> int:
    """Print or export the frozen wire contract (M51)."""
    import json
    from pathlib import Path

    from allm.wire import wire_contract

    document = json.dumps(wire_contract(), indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(document, encoding="utf-8")
        print(f"wire contract written to {args.output}")
    else:
        print(document, end="")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    """Export a standalone snapshot of the system dashboard (M50/M51)."""
    from pathlib import Path

    from allm.api.dashboard import snapshot_html, system_state
    from allm.storage import SQLiteRecordStore

    store = SQLiteRecordStore(args.db)
    try:
        if args.output:
            Path(args.output).write_text(snapshot_html(store), encoding="utf-8")
            print(f"dashboard snapshot written to {args.output}")
        else:
            import json

            print(json.dumps(system_state(store), indent=2))
    finally:
        store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="allm", description="ALLM research platform CLI")
    parser.add_argument("--log-level", default="INFO", help="logging level (default INFO)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show version and plugins").set_defaults(func=_cmd_info)

    p_config = sub.add_parser("config", help="configuration commands")
    config_sub = p_config.add_subparsers(dest="subcommand", required=True)
    p_show = config_sub.add_parser("show", help="print resolved configuration")
    p_show.add_argument("-c", "--config", default=None, help="YAML config file")
    p_show.set_defaults(func=_cmd_config_show)

    sub.add_parser("plugins", help="list registered implementations").set_defaults(
        func=_cmd_plugins
    )

    p_runs = sub.add_parser("runs", help="list experiment runs")
    p_runs.add_argument("-c", "--config", default=None, help="YAML config file")
    p_runs.set_defaults(func=_cmd_runs)

    p_model = sub.add_parser("model", help="model commands")
    model_sub = p_model.add_subparsers(dest="subcommand", required=True)
    p_validate = model_sub.add_parser("validate", help="validate a model spec YAML")
    p_validate.add_argument("spec", help="path to model spec YAML")
    p_validate.set_defaults(func=_cmd_model_validate)

    p_data = sub.add_parser("dataset", help="dataset commands")
    data_sub = p_data.add_subparsers(dest="subcommand", required=True)
    p_peek = data_sub.add_parser("peek", help="print the first samples")
    p_peek.add_argument("spec", help="path to dataset spec YAML")
    p_peek.add_argument("-n", type=int, default=3, help="samples to show (default 3)")
    p_peek.set_defaults(func=_cmd_dataset_peek)

    p_kdp = sub.add_parser("kdp", help="knowledge distillation pipeline")
    kdp_sub = p_kdp.add_subparsers(dest="subcommand", required=True)
    p_distill = kdp_sub.add_parser("distill", help="distill documents into knowledge units")
    p_distill.add_argument("files", nargs="+", help="text/markdown files to ingest")
    p_distill.add_argument(
        "--db", default=None, help="sqlite path: persist raw docs + inject into the graph"
    )
    p_distill.set_defaults(func=_cmd_kdp_distill)

    p_bench = sub.add_parser("benchmark", help="state-of-the-system report (M47)")
    p_bench.add_argument(
        "--corpora",
        default="fiction,kids,books,practice",
        help="comma-separated corpora (default: fiction,kids,books,practice)",
    )
    p_bench.add_argument("--iterations", type=int, default=3, help="loop iterations (default 3)")
    p_bench.add_argument("--seed", type=int, default=13, help="deterministic seed (default 13)")
    p_bench.add_argument("--limit", type=int, default=24, help="max samples per corpus (default 24)")
    p_bench.add_argument(
        "--student",
        default="echo",
        choices=("echo", "ollama"),
        help="student backend: echo (offline, default) or ollama (real model)",
    )
    p_bench.add_argument("--output", default=None, help="write the report JSON to this path")
    p_bench.add_argument("--root", default=None, help="project root (default: installed package root)")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_audit = sub.add_parser("audit", help="append-only write trail (M50)")
    p_audit.add_argument("--db", required=True, help="sqlite database path")
    p_audit.add_argument("-n", "--namespace", default=None, help="filter by namespace")
    p_audit.add_argument("--limit", type=int, default=50, help="rows to show (default 50)")
    p_audit.add_argument("--offset", type=int, default=0, help="rows to skip")
    p_audit.set_defaults(func=_cmd_audit)

    p_db = sub.add_parser("db", help="database maintenance (M50)")
    db_sub = p_db.add_subparsers(dest="subcommand", required=True)
    p_backup = db_sub.add_parser("backup", help="consistent online backup")
    p_backup.add_argument("--db", required=True, help="sqlite database path")
    p_backup.add_argument("destination", help="backup file to write")
    p_backup.set_defaults(func=_cmd_db_backup)
    p_restore = db_sub.add_parser("restore", help="restore a verified backup")
    p_restore.add_argument("--db", required=True, help="target database path")
    p_restore.add_argument("backup", help="backup file to restore from")
    p_restore.add_argument(
        "--force", action="store_true",
        help="replace an existing target (old file kept as .replaced)",
    )
    p_restore.set_defaults(func=_cmd_db_restore)
    p_seed = sub.add_parser("seed", help="populate a store with the public-loop scenario (M52)")
    p_seed.add_argument("--db", required=True, help="path to the SQLite store to seed")
    p_seed.add_argument("--force", action="store_true", help="seed even if the store has data")
    p_seed.set_defaults(func=_cmd_seed)

    p_wire = sub.add_parser("wire", help="print/export the frozen wire contract (M51)")
    p_wire.add_argument("--output", "-o", help="write the contract JSON (default: stdout)")
    p_wire.set_defaults(func=_cmd_wire)

    p_dash = sub.add_parser("dashboard", help="system dashboard state/snapshot (M50)")
    p_dash.add_argument("--db", required=True, help="path to the SQLite store")
    p_dash.add_argument(
        "--output", "-o", help="write a standalone HTML snapshot (default: print JSON state)"
    )
    p_dash.set_defaults(func=_cmd_dashboard)

    p_verify = db_sub.add_parser("verify", help="integrity-check a database")
    p_verify.add_argument("--db", required=True, help="sqlite database path")
    p_verify.set_defaults(func=_cmd_db_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
