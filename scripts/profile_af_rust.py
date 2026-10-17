"""Benchmark optional Rust kernels through the unchanged DeepSeek CLI."""

import argparse
from contextlib import nullcontext
import functools
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.af_rust.backend import IDENTITY, build, installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="build the Rust extension")
    parser.add_argument(
        "--offline", action="store_true", help="build from cached crates"
    )
    parser.add_argument("--backend", choices=("python", "rust"), default="rust")
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=("keys", "templates", "events", "domains"),
        default=("keys", "templates"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.build:
        build(offline=options.offline)
    arguments = options.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        if options.build:
            return
        parser.error("supply DeepSeek CLI arguments after --")

    from scripts import run_deepseek_mla as cli

    report = {
        "backend": options.backend,
        "kernels": options.kernels if options.backend == "rust" else [],
        "rust_source_sha256": IDENTITY,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": arguments,
        "phases": [],
    }
    originals = {}

    def measured(name, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                row = {"phase": name, "wall_s": perf_counter() - start}
                report["phases"].append(row)
                print(json.dumps(row), file=sys.stderr, flush=True)

        return call

    for name in ("configuration", "simulate", "write_profile"):
        originals[name] = getattr(cli, name)
        setattr(cli, name, measured(name, originals[name]))
    context = installed(options.kernels) if options.backend == "rust" else nullcontext()
    try:
        with context:
            cli.main([*arguments, "--no-resume"])
    finally:
        for name, function in originals.items():
            setattr(cli, name, function)
        resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_rss_mib"] = resident / (
            1024**2 if sys.platform == "darwin" else 1024
        )
        if options.report:
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
