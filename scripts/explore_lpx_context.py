"""Test numerical explanations of 100K output speed exceeding 10K speed."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.config.validation import number
from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import HardwareContext
from scripts.lpx.aa_model import GemmaText
from scripts.lpx.aa_search import search
from scripts.lpx.context_sensitivity import (
    MODES,
    TARGET_RATIO,
    clear_caches,
    fixed_profiles,
    generation,
    make_costs,
    point_case,
    solve_bounded,
)


def compact(point):
    rows = {row["workload"]: row for row in point["benchmarks"]}
    tps = {key: value["predicted_tps"] for key, value in rows.items()}
    return dict(
        name=point["name"],
        mode=point["mode"],
        inventory=point["inventory"],
        tps=tps,
        ratio_100k_over_10k=tps["100k"] / tps["10k"],
        reversal=tps["100k"] > tps["10k"] * (1 + 1e-10),
    )


def fixed_scan(hardware, model, workloads, settings):
    costs = make_costs(hardware, model, settings)
    try:
        result, _ = search(costs, workloads)
        points = [compact(p) for p in result["points"] if p["mode"] in MODES]
        return dict(
            settings=settings,
            points=points,
            checked=len(points),
            reversals=sum(p["reversal"] for p in points),
            maximum_ratio=max(p["ratio_100k_over_10k"] for p in points),
        )
    finally:
        clear_caches()


def evaluate(hardware, model, case, workload, settings=None, gap=0):
    costs = make_costs(hardware, model, settings)
    try:
        return generation(costs, case, workload, post_first_gap_us=gap)
    finally:
        clear_caches()


def constant_gap_fit(hardware, model, case, short, long, *, targets=(3382, 3431)):
    """Solve one shared kernel cost and one shared post-first-token stall."""
    targets = tuple(targets)
    if len(targets) != 2:
        raise ConfigurationError("the stream-gap fit requires two target speeds")
    for target in targets:
        number(target, "target TPS", positive=True)
    first = evaluate(hardware, model, case, short, {"kernel_us": 0})
    last = evaluate(hardware, model, case, long, {"kernel_us": 0})
    unit = evaluate(hardware, model, case, short, {"kernel_us": 1})
    kernel_coefficient = unit["mean_native_step_us"] - first["mean_native_step_us"]
    short_time, long_time = (1e6 / target for target in targets)
    denominator = 1 / first["output_intervals"] - 1 / last["output_intervals"]
    if denominator == 0:
        return dict(
            feasible=False,
            shared_parameters_at_both_lengths=True,
            reason="equal answer lengths cannot identify a shared overhead and shared gap",
        )
    gap = (
        short_time
        - long_time
        + last["mean_native_step_us"]
        - first["mean_native_step_us"]
    ) / denominator
    kernel = (
        short_time - first["mean_native_step_us"] - gap / first["output_intervals"]
    ) / kernel_coefficient
    result = dict(
        required_kernel_us=kernel,
        post_first_gap_us=gap,
        feasible=kernel >= 0 and gap >= 0,
        shared_parameters_at_both_lengths=True,
        explanation="different answer lengths amortize an identical post-first-token stall",
    )
    if result["feasible"]:
        settings = {"kernel_us": kernel}
        result["short"] = evaluate(hardware, model, case, short, settings, gap)
        result["long"] = evaluate(hardware, model, case, long, settings, gap)
        result["ratio"] = result["long"]["tps"] / result["short"]["tps"]
        controlled = evaluate(
            hardware, model, case, {**short, "output_tokens": 2000}, settings, gap
        )
        result["equal_output_ratio"] = result["long"]["tps"] / controlled["tps"]
    return result


def dynamic_probes(hardware, model, point, workloads):
    case = point_case(point)
    rows = {w["name"]: w for w in workloads}
    short, long = rows["10k"], rows["100k"]
    baseline_short = evaluate(hardware, model, case, short)
    baseline_long = evaluate(hardware, model, case, long)
    target = baseline_short["tps"] * TARGET_RATIO
    probes = []

    def fitted(field, short_value, bounds, increasing, target_tps, fit_workload):
        def response(value):
            return evaluate(hardware, model, case, fit_workload, {field: value})["tps"]

        value = solve_bounded(response, target_tps, *bounds, increasing=increasing)
        long_value = value if fit_workload is long else short_value
        short_value = value if fit_workload is short else short_value
        if value is None:
            probes.append(dict(parameter=field, solution=None, bounds=list(bounds)))
            return
        first = evaluate(hardware, model, case, short, {field: short_value})
        last = evaluate(hardware, model, case, long, {field: long_value})
        curve, controlled_curve = [], []
        for context in range(10000, 100001, 10000):
            fraction = (context - 10000) / 90000
            setting = short_value + fraction * (long_value - short_value)
            workload = dict(
                name=str(context),
                input_tokens=context,
                output_tokens=round(1500 + fraction * 500),
                target_tps=3400,
            )
            curve.append(
                dict(
                    context=context,
                    value=setting,
                    **evaluate(hardware, model, case, workload, {field: setting}),
                )
            )
            controlled_curve.append(
                dict(
                    context=context,
                    value=setting,
                    **evaluate(
                        hardware,
                        model,
                        case,
                        {**workload, "output_tokens": 2000},
                        {field: setting},
                    ),
                )
            )
        probes.append(
            dict(
                parameter=field,
                solution=value,
                short_value=short_value,
                long_value=long_value,
                short=first,
                long=last,
                ratio=last["tps"] / first["tps"],
                curve=curve,
                curve_monotone=all(
                    a["tps"] <= b["tps"] for a, b in zip(curve, curve[1:])
                ),
                equal_output_curve=controlled_curve,
                equal_output_curve_monotone=all(
                    a["tps"] <= b["tps"]
                    for a, b in zip(controlled_curve, controlled_curve[1:])
                ),
                interpretation="length-dependent assumption fitted to the published ratio",
            )
        )

    fitted("kernel_us", 0.2, (0, 0.2), False, target, long)
    fitted("rho_out", 1, (0.5, 1), False, target, long)
    attention_device = next(
        p.device
        for p in case["decode"]
        if p.part in {"full", "attention", "global_attention"}
    )
    full_eff = hardware.device(attention_device).analytical["memory_efficiency"]
    fitted(
        "global_attention_efficiency",
        full_eff,
        (0.001, full_eff),
        True,
        baseline_long["tps"] / TARGET_RATIO,
        short,
    )
    if point["mode"] != "lpx":
        fitted("cross_latency_us", 2, (0, 2), False, target, long)

    # One identical gap inside both output windows; prefill/TTFT is never added.
    n_short = baseline_short["output_intervals"]
    n_long = baseline_long["output_intervals"]
    gap = (
        TARGET_RATIO * baseline_long["mean_native_step_us"]
        - baseline_short["mean_native_step_us"]
    ) / (1 / n_short - TARGET_RATIO / n_long)
    first = evaluate(hardware, model, case, short, gap=gap)
    last = evaluate(hardware, model, case, long, gap=gap)
    controlled = []
    for workload in (short, long):
        controlled.append(
            evaluate(
                hardware, model, case, {**workload, "output_tokens": 2000}, gap=gap
            )
        )
    probes.append(
        dict(
            parameter="post_first_gap_us",
            solution=gap,
            short_value=gap,
            long_value=gap,
            short=first,
            long=last,
            ratio=last["tps"] / first["tps"],
            controlled_equal_output_ratio=controlled[1]["tps"] / controlled[0]["tps"],
            interpretation="same post-first-token gap amortized over different answer lengths",
        )
    )

    # Absolute two-point fit uses the published speeds, without claiming calibration.
    absolute = []
    common_profiles = [{}]
    if point["mode"] != "lpx":
        common_profiles.append({"cross_latency_us": 0.25})
    for common in common_profiles:
        fitted_rows = []
        for workload, target_tps in ((short, 3382), (long, 3431)):

            def response(value):
                return evaluate(
                    hardware, model, case, workload, {**common, "kernel_us": value}
                )["tps"]

            value = solve_bounded(response, target_tps, 0, 1, increasing=False)
            fitted_rows.append(
                dict(
                    workload=workload["name"],
                    target_tps=target_tps,
                    kernel_us=value,
                    achieved_tps=response(value) if value is not None else None,
                    zero_kernel_tps=response(0),
                )
            )
        curve = []
        if all(row["kernel_us"] is not None for row in fitted_rows):
            for context in (10000, 20000, 50000, 100000):
                fraction = (context - 10000) / 90000
                kernel = fitted_rows[0]["kernel_us"] + fraction * (
                    fitted_rows[1]["kernel_us"] - fitted_rows[0]["kernel_us"]
                )
                workload = dict(
                    input_tokens=context,
                    output_tokens=round(1500 + fraction * 500),
                )
                curve.append(
                    dict(
                        context=context,
                        kernel_us=kernel,
                        **evaluate(
                            hardware,
                            model,
                            case,
                            workload,
                            {**common, "kernel_us": kernel},
                        ),
                    )
                )
        absolute.append(dict(common_settings=common, fits=fitted_rows, curve=curve))
    return dict(
        layout=compact(point),
        probes=probes,
        absolute_kernel_fits=absolute,
        absolute_constant_gap_fit=constant_gap_fit(hardware, model, case, short, long),
        fitted_results_are_not_independent_validation=True,
    )


def adaptive_pairs(points):
    """Find changed TP/PP schedules with identical allocated chip counts."""
    result = {}
    for mode in MODES:
        rows = [p for p in points if p["mode"] == mode]
        candidates = []
        for short in rows:
            for long in rows:
                if short["inventory"] != long["inventory"]:
                    continue
                ratio = long["tps"]["100k"] / short["tps"]["10k"]
                if ratio <= 1:
                    continue
                candidates.append(
                    dict(
                        short=short,
                        long=long,
                        ratio=ratio,
                        note="TP/PP schedule changes; each static schedule is slower at 100K",
                    )
                )
        result[mode] = (
            min(candidates, key=lambda p: abs(p["ratio"] - TARGET_RATIO))
            if candidates
            else None
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/aa-gemma4.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-context-sensitivity"
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    raw = json.loads(args.config.read_text())
    model = GemmaText(**raw["model"])
    hardware = HardwareContext.load(ROOT / "data")
    args.output.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(
        args.output / "checkpoints",
        inputs=(args.config, ROOT / "data/devices", ROOT / "data/topologies"),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
        resume=not args.no_resume,
    )
    started = time.perf_counter()
    scans = []
    for name, settings in fixed_profiles():
        result = runner.run(
            dict(stage="fixed", name=name, settings=settings, config=raw),
            lambda directory: fixed_scan(hardware, model, raw["workloads"], settings),
        ).result
        scans.append(dict(name=name, **result))
        print(
            f"{name}: {result['checked']} layouts, {result['reversals']} reversals",
            flush=True,
        )

    def probes(directory):
        costs = make_costs(hardware, model)
        result, _ = search(costs, raw["workloads"])
        clear_caches()
        points = [result["selections"][mode]["best"] for mode in MODES]
        minimum = result["selections"]["lpx"]["minimum_cards"]
        if minimum and minimum["name"] != points[0]["name"]:
            points.insert(0, minimum)
        return dict(
            dynamic=[
                dynamic_probes(hardware, model, p, raw["workloads"]) for p in points
            ],
            adaptive=adaptive_pairs(scans[0]["points"]),
        )

    hypotheses = runner.run(dict(stage="dynamic", config=raw), probes).result
    report = dict(
        scope=list(MODES),
        metric="single-client generation TPS excluding TTFT",
        target_ratio=TARGET_RATIO,
        published_tps={"10k": 3382, "100k": 3431},
        fixed_profiles=scans,
        **hypotheses,
        checked=sum(row["checked"] for row in scans),
        reversals=sum(row["reversals"] for row in scans),
        wall_seconds=time.perf_counter() - started,
        limitations=[
            "Analytical sensitivity tests without real hardware calibration or new quality tests",
            "Above-catalog bandwidth/compute and zero link latency are idealized stress controls",
            "Length-dependent values are constructed fits, not evidence of actual compiler behavior",
            "Output token ratio changes the AA counting rate and native answer length",
            "Post-first gaps are hypothetical stalls inside the generation window; TTFT remains excluded",
            "Equal inventory with changed TP/PP does not hold the execution schedule constant",
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
