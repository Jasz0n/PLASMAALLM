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
        Path(args.output).write_text(report.model_dump_json(indent=2) + "\n")
        print(f"\nreport written to {args.output}")
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
