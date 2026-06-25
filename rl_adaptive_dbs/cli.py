"""``rl-dbs`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.runner import BenchmarkOptions, run_suite
from benchmarks.summary import load_results_summary, render_summary_table, write_summary_csv
from benchmarks.suite import find_repo_root, parse_controller_filter, resolve_suite_path
from rl_adaptive_dbs.config_show import format_config_text, show_config
from rl_adaptive_dbs.eval_cmd import eval_controller, records_to_summary_lines
from rl_adaptive_dbs.info import build_info_payload, format_info_text
from rl_adaptive_dbs.train_cmd import train_controller


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-dbs", description="Adaptive DBS replication tooling")
    parser.add_argument("--verbose", "-v", action="store_true", help="Extra logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Errors only")
    parser.add_argument("--seed", type=int, default=42, help="Default RNG seed")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a learned controller")
    train.add_argument("--controller", required=True)
    train.add_argument("--variant", required=True)
    train.add_argument("--seeds", help="Comma-separated training seeds")
    train.add_argument("--episodes", type=int, help="Override training episode count")
    train.add_argument("--checkpoint-dir", type=Path)
    train.add_argument("--dry-run", action="store_true")

    eval_cmd = sub.add_parser("eval", help="Evaluate a checkpoint or baseline")
    eval_cmd.add_argument("--controller", required=True)
    eval_cmd.add_argument("--variant", required=True)
    eval_cmd.add_argument("--checkpoint", type=Path)
    eval_cmd.add_argument("--seeds", help="Comma-separated eval seeds")
    eval_cmd.add_argument("--suite", help="Suite name for protocol defaults")
    eval_cmd.add_argument("--results-dir", type=Path)
    eval_cmd.add_argument("--run-id")
    eval_cmd.add_argument("--no-timeseries", action="store_true")

    benchmark = sub.add_parser("benchmark", help="Run a benchmark suite from YAML")
    benchmark.add_argument("--suite", type=Path, help="Path to suite YAML")
    benchmark.add_argument("--suite-name", help="Load suites/<name>.yaml from the project root")
    benchmark.add_argument("--results-dir", type=Path, default=Path("results"))
    benchmark.add_argument("--controllers", help="Filter controller:variant pairs")
    benchmark.add_argument("--seeds", help="Override manifest seeds")
    benchmark.add_argument("--dry-run", action="store_true")
    benchmark.add_argument("--no-timeseries", action="store_true")

    summary = sub.add_parser("summary", help="Print comparison table from results/")
    summary.add_argument("--results-dir", type=Path, default=Path("results"))
    summary.add_argument("--suite-name", help="Suite subdir under results/ (default: latest)")
    summary.add_argument("--csv", type=Path, help="Write CSV to this path")
    summary.add_argument("--width", type=int, default=100, help="Terminal table width")

    info = sub.add_parser("info", help="Repository and runtime introspection")
    info.add_argument("topic", nargs="?", help="controllers, variants, suites, env, plant, version")
    info.add_argument("--json", action="store_true")
    info.add_argument("--controller", help="Filter variants topic")

    config = sub.add_parser("config", help="Show configuration defaults")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser("show", help="Show config keys")
    config_show.add_argument("keys", nargs="*", help="Config keys (default: all)")

    return parser


def _parse_seeds(raw: str | None, default: int) -> tuple[int, ...]:
    if not raw:
        return (default,)
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_seeds_optional(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    return _parse_seeds(raw, 0)


def _check_global_flags(args: argparse.Namespace) -> int | None:
    if args.verbose and args.quiet:
        print("rl-dbs: --verbose and --quiet are mutually exclusive", file=sys.stderr)
        return 2
    return None


def _cmd_train(args: argparse.Namespace) -> int:
    try:
        summaries = train_controller(
            args.controller,
            args.variant,
            seeds=_parse_seeds(args.seeds, args.seed),
            episodes=args.episodes,
            checkpoint_dir=args.checkpoint_dir,
            dry_run=args.dry_run,
        )
    except (KeyError, NotImplementedError) as exc:
        print(f"rl-dbs train: {exc}", file=sys.stderr)
        return 3 if isinstance(exc, KeyError) else 3
    except Exception as exc:
        print(f"rl-dbs train: {exc}", file=sys.stderr)
        return 1

    if args.verbose and not args.quiet:
        for item in summaries:
            print(json.dumps(item, sort_keys=True))
    elif not args.quiet:
        label = "planned" if args.dry_run else "completed"
        print(f"train {args.controller}:{args.variant} — {len(summaries)} seed(s) {label}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    try:
        records = eval_controller(
            args.controller,
            args.variant,
            seeds=_parse_seeds(args.seeds, args.seed),
            checkpoint=args.checkpoint,
            suite_name=args.suite,
            results_dir=args.results_dir,
            run_id=args.run_id,
            write_timeseries=not args.no_timeseries,
        )
    except (KeyError, NotImplementedError) as exc:
        print(f"rl-dbs eval: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"rl-dbs eval: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"rl-dbs eval: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        for line in records_to_summary_lines(records):
            print(line)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if args.suite and args.suite_name:
        print("rl-dbs benchmark: use only one of --suite or --suite-name", file=sys.stderr)
        return 2
    if not args.suite and not args.suite_name:
        print("rl-dbs benchmark: require --suite or --suite-name", file=sys.stderr)
        return 2

    repo_root = find_repo_root()
    suite_ref = args.suite if args.suite else args.suite_name
    suite_path = resolve_suite_path(suite_ref, repo_root=repo_root)

    try:
        controller_filter = parse_controller_filter(args.controllers)
    except ValueError as exc:
        print(f"rl-dbs benchmark: {exc}", file=sys.stderr)
        return 2

    options = BenchmarkOptions(
        results_dir=args.results_dir,
        seeds=_parse_seeds_optional(args.seeds),
        controller_filter=controller_filter,
        dry_run=args.dry_run,
        write_timeseries=not args.no_timeseries,
        repo_root=repo_root,
    )
    result = run_suite(suite_path, options=options)

    if args.verbose and not args.quiet:
        print(f"suite: {result.suite.name} ({result.suite.protocol})")
        print(f"manifest: {result.manifest_path}")
        print(f"planned runs: {len(result.planned)}")
        if args.dry_run:
            for item in result.planned:
                ckpt = f" checkpoint={item.checkpoint}" if item.checkpoint else ""
                print(f"  - {item.controller}:{item.variant} seed={item.seed}{ckpt}")
        else:
            print(f"completed runs: {len(result.records)}")
            for record in result.records:
                metrics = record.metrics
                print(
                    f"  - {record.planned.controller}:{record.planned.variant} "
                    f"seed={record.planned.seed} "
                    f"p_beta_mean={metrics['p_beta_mean']:.2f} "
                    f"-> {record.run_dir.name}"
                )
    elif not args.quiet:
        action = "planned" if args.dry_run else "completed"
        print(
            f"{result.suite.name}: {len(result.planned)} runs {action}; "
            f"manifest {result.manifest_path}"
        )
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    try:
        suite = load_results_summary(args.results_dir, args.suite_name)
    except FileNotFoundError as exc:
        print(f"rl-dbs summary: {exc}", file=sys.stderr)
        return 4
    if args.csv:
        write_summary_csv(args.csv, suite)
        if not args.quiet:
            print(f"wrote {args.csv}")
    if not args.quiet or not args.csv:
        print(render_summary_table(suite, width=args.width))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    try:
        payload = build_info_payload(
            args.topic,
            controller=args.controller,
            repo_root=find_repo_root(),
        )
    except KeyError as exc:
        print(f"rl-dbs info: {exc}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_info_text(payload))
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    if args.config_command != "show":
        print("rl-dbs config: only 'show' is implemented", file=sys.stderr)
        return 2
    try:
        payload = show_config(args.keys or None)
    except KeyError as exc:
        print(f"rl-dbs config: {exc}", file=sys.stderr)
        return 3
    print(format_config_text(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    code = _check_global_flags(args)
    if code is not None:
        return code

    handlers = {
        "train": _cmd_train,
        "eval": _cmd_eval,
        "benchmark": _cmd_benchmark,
        "summary": _cmd_summary,
        "info": _cmd_info,
        "config": _cmd_config,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error(f"unknown command {args.command!r}")
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
