"""Evaluate conventional AF communication and context-speed hypotheses."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import HardwareContext
from scripts.explore_lpx_context import compact, dynamic_probes, evaluate
from scripts.lpx.aa_model import GemmaText
from scripts.lpx.aa_search import evaluate_case, replay, search
from scripts.lpx.context_sensitivity import (
    clear_caches,
    make_costs,
    point_case,
    solve_bounded,
)


def profiles():
    yield "baseline", {}
    for kernel in (0, 0.02, 0.05, 0.1, 0.2):
        for latency in (0, 0.025, 0.05, 0.1, 0.25, 0.5, 1):
            if kernel == 0.2 and latency == 1:
                continue
            yield f"kernel-{kernel}-latency-{latency}", dict(
                kernel_us=kernel, link_latency_scale=latency
            )
    for latency in (0, 0.25, 0.5, 1):
        yield f"cross-startup-{latency}", dict(cross_latency_us=latency)
    yield "cross-and-kernel-zero", dict(kernel_us=0, cross_latency_us=0)


def scan(hardware, model, workloads, settings, baseline):
    costs = make_costs(hardware, model, settings)
    try:
        result, _ = search(costs, workloads)
        points = [p for p in result["points"] if p["mode"] == "af"]
        if not points:
            raise ConfigurationError(
                "no conventional AF layout fits the configured model"
            )
        best = result["selections"]["af"]["best"]
        fixed = evaluate_case(costs, point_case(baseline or best), workloads)
        return dict(
            settings=settings,
            best=best,
            minimum_cards=result["selections"]["af"]["minimum_cards"],
            baseline_inventory=fixed,
            points=[compact(p) for p in points],
            checked=len(points),
            reversals=sum(compact(p)["reversal"] for p in points),
        )
    finally:
        clear_caches()


def fit_latency(hardware, model, point, workloads):
    """Fit changing effective startup at fixed physical inventory and bandwidth."""
    case = point_case(point)
    rows = {w["name"]: w for w in workloads}
    shared = {"kernel_us": 0.02}
    fits = []
    for name, target in (("10k", 3382), ("100k", 3431)):

        def response(value):
            return evaluate(
                hardware,
                model,
                case,
                rows[name],
                {**shared, "link_latency_scale": value},
            )["tps"]

        scale = solve_bounded(response, target, 0, 1, increasing=False)
        fits.append(
            dict(
                workload=name,
                target_tps=target,
                link_latency_scale=scale,
                achieved_tps=response(scale) if scale is not None else None,
                zero_startup_tps=response(0),
            )
        )
    curves = []
    replays = {}
    if all(fit["link_latency_scale"] is not None for fit in fits):
        for context in range(10000, 100001, 10000):
            fraction = (context - 10000) / 90000
            scale = fits[0]["link_latency_scale"] + fraction * (
                fits[1]["link_latency_scale"] - fits[0]["link_latency_scale"]
            )
            settings = {**shared, "link_latency_scale": scale}
            workload = dict(
                name=str(context),
                input_tokens=context,
                output_tokens=round(1500 + fraction * 500),
            )
            curves.append(
                dict(
                    context=context,
                    link_latency_scale=scale,
                    **evaluate(hardware, model, case, workload, settings),
                    equal_output_tps=evaluate(
                        hardware,
                        model,
                        case,
                        {**workload, "output_tokens": 2000},
                        settings,
                    )["tps"],
                )
            )
        for fit in fits:
            costs = make_costs(
                hardware,
                model,
                {**shared, "link_latency_scale": fit["link_latency_scale"]},
            )
            try:
                replays[fit["workload"]] = replay(
                    costs, case, [rows[fit["workload"]]], requests=50, warmup=5
                )
            finally:
                clear_caches()
    return dict(
        layout=compact(point),
        shared_settings=shared,
        fits=fits,
        curves=curves,
        curve_monotone=bool(curves)
        and all(a["tps"] <= b["tps"] for a, b in zip(curves, curves[1:])),
        equal_output_curve_monotone=bool(curves)
        and all(
            a["equal_output_tps"] <= b["equal_output_tps"]
            for a, b in zip(curves, curves[1:])
        ),
        replays=replays,
        interpretation="length-dependent effective startup fit without hardware calibration",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/aa-gemma4.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-regular-af-20261019"
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    raw = json.loads(args.config.read_text())
    names = {w["name"] for w in raw["workloads"]}
    if not {"10k", "100k"} <= names:
        raise ConfigurationError(
            "conventional AF comparison requires 10k and 100k workloads"
        )
    hardware = HardwareContext.load(ROOT / "data")
    model = GemmaText(**raw["model"])
    args.output.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(
        args.output / "checkpoints",
        inputs=(args.config, ROOT / "data/devices", ROOT / "data/topologies"),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts"),
        resume=not args.no_resume,
    )
    started = time.perf_counter()
    scans = []
    baseline = None
    for name, settings in profiles():
        result = runner.run(
            dict(stage="constant", name=name, settings=settings, config=raw),
            lambda directory: scan(
                hardware, model, raw["workloads"], settings, baseline
            ),
        ).result
        scans.append(dict(name=name, **result))
        baseline = baseline or result["best"]
        if len(scans) % 5 == 0 or name == "baseline":
            print(f"Completed {len(scans)} profiles; latest {name}", flush=True)

    def hypotheses(directory):
        optimistic = next(
            p["best"] for p in scans if p["name"] == "kernel-0.02-latency-0"
        )
        costs = make_costs(hardware, model)
        try:
            baseline_replay = replay(
                costs, point_case(baseline), raw["workloads"], requests=50, warmup=5
            )
        finally:
            clear_caches()
        return dict(
            baseline_hypotheses=dynamic_probes(
                hardware, model, baseline, raw["workloads"]
            ),
            length_dependent_startup=fit_latency(
                hardware, model, optimistic, raw["workloads"]
            ),
            baseline_replay=baseline_replay,
        )

    hypothesis = runner.run(dict(stage="hypotheses", config=raw), hypotheses).result
    sources = [
        Path(__file__),
        ROOT / "scripts/explore_lpx_context.py",
        *sorted((ROOT / "scripts/lpx").glob("*.py")),
        args.config.resolve(),
        ROOT / "data/devices/rubin.yaml",
        ROOT / "data/devices/groq_lp30.yaml",
        ROOT / "data/topologies/links.yaml",
    ]
    report = dict(
        schema_version=1,
        scope="conventional AF: all attention on Rubin, all FFNs on LPX",
        result_class="analytical sensitivity, not physical hardware measurements",
        metric="single-client generation TPS excluding TTFT",
        workloads=raw["workloads"],
        profiles=scans,
        **hypothesis,
        checked=sum(p["checked"] for p in scans),
        reversals=sum(p["reversals"] for p in scans),
        wall_seconds=time.perf_counter() - started,
        source_sha256={
            str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sources
        },
        limitations=[
            "No cross-request concurrency, generated text, statistical samples, or quality tests",
            "Baseline prefill and all decode attention share Rubin; there is no KV migration to LPX",
            "Every output token pays AF activation-transfer and broadcast costs at every layer",
            "TTFT and prefill are excluded; hypothetical stream gaps must occur after first output",
            "Reduced startup profiles are assumptions, including zero-latency stress controls",
            "The startup fit changes the effective communication model with context length",
            "Different selected layouts across constant profiles do not establish a fixed-inventory speedup",
            "Scheduled direct collectives and serial compute/communication composition are uncalibrated",
        ],
    )
    target = args.output / "summary.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Saved {target}; {report['checked']} layouts; {report['reversals']} fixed-setting reversals",
        flush=True,
    )


if __name__ == "__main__":
    main()
