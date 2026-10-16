"""Run the historical 32-user V3 workload using this checkout's AF runtime."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.user_qps.configuration import historical_runtime
from scripts.user_qps.runner import dump, experiment, run_case


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qps",
        type=float,
        nargs="+",
        default=[4.0],
        help="Aggregate Poisson arrival rates, in requests/s; run sequentially.",
    )
    parser.add_argument("--users", type=int, default=32)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument(
        "--unique-users",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use distinct shuffled user IDs, one request per selected user.",
    )
    parser.add_argument("--output-tokens", type=int, default=400)
    parser.add_argument(
        "--context",
        type=int,
        default=2048,
        help="Initial resident KV tokens per request; prefill is excluded.",
    )
    parser.add_argument("--a-tp", type=int, choices=[4], default=4)
    parser.add_argument("--f-ep", type=int, choices=[32], default=32)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--af-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--busy-policy", choices=["queue"], default="queue")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing directories are rejected.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export and validate the configuration and arrivals without executing decode.",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    counts = (
        args.users,
        args.requests,
        args.output_tokens,
        args.context,
        args.max_batch,
        args.af_batch_size,
    )
    if min(counts) < 1 or any(not math.isfinite(q) or q <= 0 for q in args.qps):
        parser.error("rates and counts must be positive and finite")
    if len({f"{q:g}" for q in args.qps}) != len(args.qps):
        parser.error("duplicate QPS points or colliding output directory names")
    if args.unique_users and args.requests > args.users:
        parser.error("--unique-users requires --requests <= --users")
    if args.output.exists():
        parser.error("output directory already exists; select a new --output")
    return args


def main(argv=None):
    args = parse_args(argv)
    runtime = historical_runtime(args)
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "experiment.json", experiment(args, runtime))
    rows = []
    for qps in args.qps:
        row = run_case(args, runtime, qps)
        if row is not None:
            rows.append(row)
            dump(args.output / "results.json", rows)
            with (args.output / "results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                fields = [name for name in rows[0] if name != "operator_stats"]
                writer = csv.DictWriter(
                    stream, fieldnames=fields, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
