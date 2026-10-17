"""Profile the complete production AF probe, separating generated record methods.

Run with --steps N --output DIRECTORY; the existing AF probe measures phases,
hashes all public JSON outputs and removes its large temporary trace afterward.
runtime.prof covers cli.main only, excluding imports, hashing and cleanup.
Use --mode sample for a lower-overhead wall-clock stack sample instead.
"""

import argparse
from collections import defaultdict
import cProfile
from dataclasses import is_dataclass
import gc
import json
from pathlib import Path
import pstats
import runpy
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_deepseek_mla as cli
from scripts.af_probe_samples import StackSampler
from TokenSim.af import factory


def identify_generated_methods():
    """Give dataclass code distinct profiler identities without wrapping calls."""
    changed = []
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("TokenSim.") or module is None:
            continue
        for record in tuple(vars(module).values()):
            if (
                not isinstance(record, type)
                or record.__module__ != module_name
                or not is_dataclass(record)
            ):
                continue
            for name, method in vars(record).items():
                code = getattr(method, "__code__", None)
                if code is None or code.co_filename != "<string>":
                    continue
                changed.append((method, code))
                filename = f"<generated:{module_name}.{record.__qualname__}.{name}>"
                method.__code__ = code.replace(co_filename=filename)
    return changed


def category(key):
    filename, _, name = key
    path = filename.replace("\\", "/")
    if filename.startswith("<generated:"):
        return "trace_capture" if ".AFTraceEvent." in filename else "record_methods"
    if "/json/" in path or "af/trace/export.py" in path:
        return "trace_json"
    if "af/trace/pipeline.py" in path and name in (
        "append",
        "_prefix",
        "_writer",
        "finish",
    ):
        return "trace_json"
    if "af/trace/recorder.py" in path and name in (
        "validate_invariants",
        "_validate_runtime_layer_semantics",
    ):
        return "trace_validation"
    if "af/trace/metrics.py" in path or "af/trace/occupancy.py" in path:
        return "trace_validation"
    if any(
        part in path
        for part in ("af/trace/archive", "af/trace/history.py", "af/trace/pipeline.py")
    ):
        return "trace_archive"
    if "af/trace/" in path or ".AFTraceEvent." in path:
        return "trace_capture"
    if any(part in path for part in ("af/caching/estimates.py", "af/providers/")):
        return "queries_and_estimates"
    if any(part in path for part in ("af/caching/templates.py", "af/planning/")):
        return "dag_planning"
    if "af/communication/resources.py" in path:
        return "resource_scheduler"
    if "/simpy/" in path:
        return "simpy_engine"
    if "af/execution/" in path:
        return "execution"
    if "/TokenSim/" in path or "/scripts/" in path:
        return "other_project"
    return None


def summarize(profile_path):
    """Partition self time once, assigning generic helpers by direct caller.

    pstats caller edge self times attribute builtins and standard helpers to
    their direct callers; helpers called by another generic helper stay in
    python_runtime instead of guessing a whole-stack ownership.
    """
    stats = pstats.Stats(str(profile_path))
    totals, rows = defaultdict(float), []
    for key, (primitive, calls, own, cumulative, callers) in stats.stats.items():
        group = category(key)
        allocation = defaultdict(float)
        if group is not None:
            allocation[group] = own
        else:
            for caller, edge in callers.items():
                edge_self = edge[2] if isinstance(edge, tuple) else 0.0
                allocation[category(caller) or "python_runtime"] += edge_self
            accounted = sum(allocation.values())
            if accounted > 0:
                allocation = {
                    label: seconds * own / accounted
                    for label, seconds in allocation.items()
                }
            else:
                allocation["python_runtime"] = own
        for label, seconds in allocation.items():
            totals[label] += seconds
        rows.append(
            dict(
                file=key[0].replace(str(ROOT) + "/", ""),
                line=key[1],
                function=key[2],
                primitive_calls=primitive,
                calls=calls,
                self_seconds=own,
                cumulative_seconds=cumulative,
                category=group,
                self_allocation=dict(allocation),
            )
        )
    return dict(
        total_self_seconds=stats.total_tt,
        categories={
            label: dict(seconds=seconds, percent=100 * seconds / stats.total_tt)
            for label, seconds in sorted(
                totals.items(), key=lambda item: item[1], reverse=True
            )
        },
        functions=sorted(rows, key=lambda row: row["self_seconds"], reverse=True),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--mode", choices=("calls", "sample"), default="calls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile_path = args.output / "runtime.prof"
    original_main = cli.main
    original_run = factory.run_environment
    changed = identify_generated_methods()
    collections, pending = [], {}
    sampled = None
    step_samples = []

    def sampled_run(*values, **options):
        if sampled is None:
            return original_run(*values, **options)
        sampled._sample(None, sys._getframe())
        before = dict(sampled.totals)
        try:
            return original_run(*values, **options)
        finally:
            sampled._sample(None, sys._getframe())
            step_samples.append(
                dict(
                    step=len(step_samples),
                    categories={
                        group: seconds - before.get(group, 0)
                        for group, seconds in sampled.totals.items()
                    },
                )
            )

    def collection(phase, info):
        generation = info["generation"]
        if phase == "start":
            pending[generation] = perf_counter()
        elif generation in pending:
            collections.append(
                dict(
                    generation=generation,
                    seconds=perf_counter() - pending.pop(generation),
                    collected=info["collected"],
                )
            )

    def profiled_main(values):
        nonlocal sampled
        if args.mode == "sample":
            sampled = StackSampler(category)
            with sampled:
                return original_main(values)
        profile = cProfile.Profile()
        gc.callbacks.append(collection)
        try:
            with profile:
                return original_main(values)
        finally:
            gc.callbacks.remove(collection)
            profile.dump_stats(str(profile_path))

    cli.main = profiled_main
    if args.mode == "sample":
        factory.run_environment = sampled_run
    previous_argv = sys.argv
    sys.argv = [
        "profile_af_probe",
        "--steps",
        str(args.steps),
        "--output",
        str(args.output),
    ]
    try:
        runpy.run_module("scripts.profile_af_probe", run_name="__main__")
    finally:
        sys.argv = previous_argv
        cli.main = original_main
        factory.run_environment = original_run
        for method, code in changed:
            method.__code__ = code
    result = sampled.summarize() if sampled is not None else summarize(profile_path)
    result["mode"] = args.mode
    if sampled is not None:
        result["steps"] = step_samples
    result["generated_methods_identified"] = len(changed)
    if sampled is None:
        result["garbage_collections"] = collections
        result["gc_seconds"] = sum(item["seconds"] for item in collections)
    (args.output / "breakdown.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("categories", "gc_seconds")}))


if __name__ == "__main__":
    main()
