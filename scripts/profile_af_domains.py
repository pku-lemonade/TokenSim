"""Isolate resource scheduling costs; complete-run timings remain the adoption gate."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simpy

from TokenSim.af.communication.resources import DomainScheduler
from TokenSim.af.runtime.host import host_collection
from scripts.af_rust import backend
from scripts.af_rust.domains import scheduler_type


def measure(kind, work, width):
    env = simpy.Environment()
    scheduler = kind(env, tuple(f"resource:{i}" for i in range(32)))
    rows = [None] * len(work)

    def service(index):
        keys, duration = work[index]
        rows[index] = yield from scheduler.run(keys, duration, (index,))

    def batches():
        for start in range(0, len(work), width):
            yield env.all_of(
                [
                    env.process(service(index))
                    for index in range(start, min(start + width, len(work)))
                ]
            )

    env.process(batches())
    gc.collect()
    with host_collection():
        started = perf_counter()
        env.run()
        elapsed = perf_counter() - started
    if scheduler.owners or scheduler.waiters:
        raise RuntimeError("scheduler retained active services")
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return {"wall_seconds": elapsed, "interval_sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=16384)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.jobs, args.width, args.repeats) < 1:
        parser.error("jobs, width and repeats must be positive")
    rng = random.Random(42)
    keys = tuple(f"resource:{i}" for i in range(32))
    work = [
        (tuple(rng.sample(keys, rng.randint(1, 3))), rng.randint(1, 5) / 1e6)
        for _ in range(args.jobs)
    ]
    kinds = {
        "python": DomainScheduler,
        "rust": scheduler_type(backend.library()),
    }
    report = {
        "jobs": args.jobs,
        "width": args.width,
        "python_version": sys.version,
        "rust_source_sha256": backend.IDENTITY,
        "rows": [],
    }
    for repeat in range(args.repeats):
        order = ("python", "rust") if repeat % 2 == 0 else ("rust", "python")
        for name in order:
            row = dict(
                backend=name, repeat=repeat, **measure(kinds[name], work, args.width)
            )
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    if len({row["interval_sha256"] for row in report["rows"]}) != 1:
        raise RuntimeError("native service intervals differ from the reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
