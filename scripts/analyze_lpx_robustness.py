"""Check SRAM reserves and execution-cost boundaries of saved LPX candidates."""

import argparse
from dataclasses import replace
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import DeviceCatalog, HardwareContext
from TokenSim.hardware.device.sourced_value import SourcedValue
from scripts.lpx.config import Layout
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.native_config import configuration
from scripts.lpx.native_search import memory_report, screen
from scripts.lpx.partition import native_footprint, partition_solutions
from scripts.lpx.replay_profiles import source_identity


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reserved_hardware(hardware, extra_bytes):
    """Reserve additional physical SRAM without changing execution rates."""
    return replace(
        hardware,
        devices=DeviceCatalog(
            (
                replace(
                    device,
                    memory_reserved_bytes=SourcedValue(
                        device.memory_reserved_bytes.value + extra_bytes,
                        "experiment:additional-unmeasured-sram-reserve",
                        "D",
                    ),
                )
                if device.device_id == "groq_lp30"
                else device
            )
            for device in hardware.devices
        ),
        _packages=dict(hardware._packages),
    )


def layout_for(tp, pp):
    return Layout(
        a_device="groq_lp30",
        f_device="groq_lp30",
        a_tp=tp,
        f_tp=tp,
        a_pp=pp,
        f_pp=pp,
        f_ep=1,
    )


def lookup_rows(data):
    return {
        (row["layout"]["a_tp"], row["layout"]["a_pp"]): row
        for row in data["rows"]
        if row["status"] == "ok"
    }


def selected(row):
    if row is None:
        return None
    return dict(
        tp=row["layout"]["a_tp"],
        pp=row["layout"]["a_pp"],
        chips=row["lp30"],
        counts=row["pipeline_layer_counts"],
        generation_tps=[w["generation_tps"] for w in row["workloads"]],
        acceptance=row["acceptance"],
    )


def reserve_matrix(hardware, model, baseline):
    saved, footprints = lookup_rows(baseline), {}
    for tp in (8, 16, 32):
        pp = min(60, 256 // tp)
        runtime = configuration(
            hardware,
            model,
            layout_for(tp, pp),
            namespace="reserve",
            colocated_dense_reuse=True,
        )
        footprints[tp] = native_footprint(runtime, 102000)
    rows = []
    for chunk in (1, 4, 16):
        for reserve in (
            0,
            5000000,
            10000000,
            20000000,
            30000000,
            50000000,
            75000000,
            100000000,
        ):
            choices, admitted = [], []
            adjusted = reserved_hardware(hardware, reserve)
            for tp, footprint in footprints.items():
                budget = replace(
                    footprint,
                    buffers=tuple(n * chunk for n in footprint.buffers),
                    capacities=tuple(n - reserve for n in footprint.capacities),
                )
                solutions = partition_solutions(budget, min(60, 256 // tp))
                for pp, partition in solutions.items():
                    point = saved[tp, pp]
                    assert (
                        tuple(point["pipeline_layer_counts"])
                        == partition["layer_counts"]
                    )
                    choices.append(point)
                if not solutions:
                    continue
                pp = min(solutions)
                counts = solutions[pp]["layer_counts"]
                runtime = configuration(
                    adjusted,
                    model,
                    layout_for(tp, pp),
                    namespace="reserve",
                    micro_batch=chunk,
                    colocated_dense_reuse=True,
                    pipeline_layer_counts=counts,
                )
                assert native_footprint(runtime, 102000) == budget
                admitted.append(
                    dict(
                        tp=tp,
                        pp=pp,
                        chips=tp * pp,
                        counts=counts,
                        memory=memory_report(runtime, 102000),
                    )
                )
            result = dict(
                prefill_chunk=chunk,
                extra_reserve_bytes=reserve,
                feasible_points=len(choices),
                native_memory_minima=admitted,
            )
            for target in ("common_3400", "published_targets"):
                passing = [row for row in choices if row["acceptance"][target]]
                minimum = min(
                    passing,
                    key=lambda row: (row["lp30"], -row["acceptance"]["common_margin"]),
                    default=None,
                )
                result[target] = selected(minimum)
                if minimum is not None:
                    runtime = configuration(
                        adjusted,
                        model,
                        Layout(**minimum["layout"]),
                        namespace="reserve",
                        micro_batch=chunk,
                        colocated_dense_reuse=True,
                        pipeline_layer_counts=minimum["pipeline_layer_counts"],
                    )
                    result[target]["native_memory"] = memory_report(runtime, 102000)
            rows.append(result)
            print(
                json.dumps(
                    dict(
                        stage="reserve",
                        chunk=chunk,
                        reserve=reserve,
                        minimum=selected(minimum),
                    )
                ),
                flush=True,
            )
    return rows


def execution_boundaries(low, high):
    result, upper = [], lookup_rows(high)
    for row in low["rows"]:
        if row["status"] != "ok" or (
            row["layout"]["a_tp"],
            row["layout"]["a_pp"],
        ) not in ((8, 11), (16, 7)):
            continue
        other = upper[row["layout"]["a_tp"], row["layout"]["a_pp"]]
        assert row["pipeline_layer_counts"] == other["pipeline_layer_counts"]
        workloads = []
        for first, second in zip(row["workloads"], other["workloads"]):
            delta = (second["generation_seconds"] - first["generation_seconds"]) / (
                first["output_tokens"] - 1
            )
            slope = delta / 0.05
            assert slope > 0
            duration = 1 / first["generation_tps"]
            target = first["published_target_tps"]
            workloads.append(
                dict(
                    input_tokens=first["input_tokens"],
                    output_tokens=first["output_tokens"],
                    target_tps=target,
                    base_seconds_per_step=duration,
                    seconds_per_instruction_us=slope,
                    instruction_limit_us=0.05 + (1 / target - duration) / slope,
                    invocation_limit_us=1 + (1 / target - duration) * 1e6,
                )
            )
        result.append(
            dict(
                candidate=selected(row),
                workloads=workloads,
                estimated_instruction_limit_us=min(
                    w["instruction_limit_us"] for w in workloads
                ),
            )
        )
    assert {r["candidate"]["chips"] for r in result} == {88, 112}
    return result


def verify_boundaries(hardware, model, rows, runner):
    for row in rows:
        candidate = row["candidate"]
        checks = []
        limit = row["estimated_instruction_limit_us"]
        for delta in (-0.0001, 0.0001):
            floor = limit + delta
            assumptions = ExecutionAssumptions(
                mode="compiled",
                instruction_floor_us=floor,
                c2c_hop_us=0.35,
                invocation_us=1,
            )
            adjusted = with_execution(hardware, assumptions)
            layout = layout_for(candidate["tp"], candidate["pp"])
            parameters = dict(
                layout=candidate, execution=assumptions.report(), requests=50, warmup=5
            )

            def execute(directory):
                return screen(
                    adjusted,
                    model,
                    layout,
                    verify=True,
                    invocation_us=1,
                    layer_counts=candidate["counts"],
                )

            receipt = runner.run(parameters, execute)
            measured = receipt.result
            assert measured["status"] == "ok"
            assert measured["acceptance"]["published_targets"] == (delta < 0)
            for work, base in zip(measured["workloads"], row["workloads"]):
                expected = (
                    base["base_seconds_per_step"]
                    + (floor - 0.05) * base["seconds_per_instruction_us"]
                )
                assert isclose(1 / work["generation_tps"], expected, rel_tol=1e-9)
            checks.append(
                dict(
                    instruction_floor_us=floor,
                    passed=delta < 0,
                    generation_tps=[w["generation_tps"] for w in measured["workloads"]],
                    checkpoint_key=receipt.key,
                    result_file=str(receipt.directory / "result.json"),
                )
            )
            print(
                json.dumps(
                    dict(stage="boundary", chips=candidate["chips"], **checks[-1])
                ),
                flush=True,
            )
        row["native_limit_bracket_us"] = [c["instruction_floor_us"] for c in checks]
        row["native_serial_checks"] = checks


def verify_reserved_request(hardware, model, matrix, runner):
    from scripts.validate_lpx_placement import complete_request

    setting = next(
        row
        for row in matrix
        if row["prefill_chunk"] == 16 and row["extra_reserve_bytes"] == 50000000
    )
    minimum = setting["published_targets"]
    parameters = dict(
        kind="complete-reserved-request",
        minimum=minimum,
        extra_reserve_bytes=setting["extra_reserve_bytes"],
        execution=ExecutionAssumptions(
            mode="compiled", instruction_floor_us=0.05, c2c_hop_us=0.35, invocation_us=1
        ).report(),
    )

    def execute(directory):
        return complete_request(
            reserved_hardware(hardware, setting["extra_reserve_bytes"]),
            model,
            minimum["tp"],
            minimum["pp"],
            16,
            100000,
            2000,
            full=True,
        )

    receipt = runner.run(parameters, execute)
    result = receipt.result
    assert result["full_prefill_executed"]
    assert result["pipeline_layer_counts"] == minimum["counts"]
    assert isclose(result["generation_tps"], minimum["generation_tps"][2], rel_tol=1e-9)
    print(json.dumps(dict(stage="reserved-full-request", result=result)), flush=True)
    return dict(
        extra_reserve_bytes=setting["extra_reserve_bytes"],
        result=result,
        checkpoint_key=receipt.key,
        result_file=str(receipt.directory / "result.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpx-robustness")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "experiments/lpx/robustness-results.json"
    )
    args = parser.parse_args()
    started = perf_counter()
    paths = [
        args.baseline / name / "results.json"
        for name in ("capacity-floor-005", "capacity-floor-010")
    ]
    low, high = [json.loads(path.read_text()) for path in paths]
    assert low["model"] == high["model"]
    for data, floor in ((low, 0.05), (high, 0.1)):
        execution = data["settings"]["execution"]
        assert (
            execution["mode"] == "compiled"
            and execution["instruction_floor_us"] == floor
        )
        assert execution["c2c_hop_us"] == 0.35 and execution["invocation_us"] == 1
        assert data["settings"]["pipeline_partition"] == "capacity"
    assumptions = ExecutionAssumptions(
        mode="compiled", instruction_floor_us=0.05, c2c_hop_us=0.35, invocation_us=1
    )
    hardware = with_execution(HardwareContext.load(ROOT / "data"), assumptions)
    model = replace(
        hardware.model("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
    )
    assert model.to_dict() == low["model"]
    result = dict(
        schema="tokensim.lpx_conditional_robustness",
        version=1,
        source_sha256=source_identity(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        execution=assumptions.report(),
        input_results=[
            dict(
                path=str(p),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                source_sha256=d["source_sha256"],
            )
            for p, d in zip(paths, (low, high))
        ],
        limitations=[
            "All reserves and execution fixed costs are sensitivity values without hardware calibration.",
            "Memory sweeps cover TP8/16/32, capacity-minimizing contiguous PP, fixed endpoint embeddings, and one serial client.",
            "Decode speeds in the memory matrix reuse exact-layout saved native samples; reserves do not change service rates.",
            "The prefill buffers are configured protocol buffers; additional reserves do not certify any particular kernel's workspace.",
            "Instruction boundary estimates assume affine service costs; fresh native serial probes check both sides of each reported bracket.",
        ],
    )
    result["reserve_matrix"] = reserve_matrix(hardware, model, low)
    result["execution_boundaries"] = execution_boundaries(low, high)
    write_json(args.output / "progress.json", result)
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data", *paths),
        code_paths=(
            ROOT / "TokenSim",
            ROOT / "scripts/lpx",
            Path(__file__),
            ROOT / "scripts/validate_lpx_placement.py",
        ),
    )
    verify_boundaries(hardware, model, result["execution_boundaries"], runner)
    result["reserved_complete_request"] = verify_reserved_request(
        hardware, model, result["reserve_matrix"], runner
    )
    result["host_seconds"] = perf_counter() - started
    write_json(args.output / "results.json", result)
    write_json(args.report, result)


if __name__ == "__main__":
    main()
