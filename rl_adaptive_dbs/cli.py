"""``rl-dbs`` command-line interface (Phase 4 — ``benchmark`` first)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.runner import BenchmarkOptions, run_suite
from benchmarks.suite import find_repo_root, parse_controller_filter, resolve_suite_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl-dbs", description="Adaptive DBS replication tooling")
    parser.add_argument("--verbose", "-v", action="store_true", help="Extra logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Errors only")
    sub = parser.add_subparsers(dest="command", required=True)

    benchmark = sub.add_parser("benchmark", help="Run a benchmark suite from YAML")
    benchmark.add_argument("--suite", type=Path, help="Path to suite YAML")
    benchmark.add_argument(
        "--suite-name",
        help="Load suites/<name>.yaml from the project root",
    )
    benchmark.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Output root (default: results/)",
    )
    benchmark.add_argument(
        "--controllers",
        help="Filter runs: controller:variant pairs, comma-separated",
    )
    benchmark.add_argument(
        "--seeds",
        help="Override manifest seed list, comma-separated integers",
    )
    benchmark.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifest only; do not execute rollouts",
    )
    benchmark.add_argument(
        "--no-timeseries",
        action="store_true",
        help="Skip writing timeseries/ per run",
    )
    return parser


def _parse_seeds(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


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
        seeds=_parse_seeds(args.seeds),
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
