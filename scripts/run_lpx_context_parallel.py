"""Run fixed-layout KV-parallel (Helix-style) LPX decode experiments.

Every case uses the integrated Gemma 4 31B TP16/PP8 layout on 128 LP30 chips
with the bundled fidelity configuration, and changes only the cases' stated
KV-parallel, ownership or communication setting.  See
docs/lpx-context-parallel.md for the specification and recorded results.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.errors import ConfigurationError
from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.fidelity import load, with_fidelity
from scripts.lpx.native_search import screen
from scripts.lpx.partition import capacity_partition
from scripts.lpx.replay_profiles import source_identity
from scripts.search_lpx_native import write_json

TP, PP, LAYER_COUNTS = 16, 8, [6, 8, 9, 7, 7, 10, 7, 6]
MAXIMUM_CONTEXT = 102000

CASES = {
    "kvp1": dict(kvp=1),
    "kvp2": dict(kvp=2),
    "kvp4": dict(kvp=4),
    # One block of ceil(maximum_context / KVP) positions per rank.
    "kvp4-contiguous": dict(kvp=4, block=-(-MAXIMUM_CONTEXT // 4)),
    "kvp4-block64": dict(kvp=4, block=64),
    # Replicated windowed-layer projections exceed the baseline partition, so
    # this case uses the PP8 capacity partition for all-layer KVP4.
    "kvp4-all-layers": dict(kvp=4, layers="all", layer_counts=[6, 8, 8, 8, 8, 8, 8, 6]),
    "kvp1-launch0.5": dict(kvp=1, launch_us=0.5),
    "kvp4-launch0.5": dict(kvp=4, launch_us=0.5),
    "kvp1-local0.7": dict(kvp=1, fabric=dict(local_us=0.7)),
    "kvp4-local0.7": dict(kvp=4, fabric=dict(local_us=0.7)),
}


def environment(config, fabric=None):
    execution = ExecutionAssumptions(
        mode="compiled", instruction_floor_us=0, c2c_hop_us=0.35
    )
    if fabric:
        config = {**config, "fabric": {**config["fabric"], **fabric}}
    hardware = with_fidelity(
        with_execution(HardwareContext.load(ROOT / "data"), execution), config
    )
    model = replace(
        hardware.models.resolve("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
    )
    return hardware, model, execution, config


def layout_for(kvp, pp=PP, tp=TP):
    return Layout(
        a_device="groq_lp30",
        f_device="groq_lp30",
        a_tp=tp // kvp,
        a_cp=kvp,
        f_tp=tp,
        a_pp=pp,
        f_pp=pp,
        f_ep=1,
    )


def kv_settings(case):
    if case["kvp"] == 1:
        return None
    return dict(
        kv_parallel_layers=case.get("layers", "full_attention"),
        kv_append_block=case.get("block", 16),
    )


def run_case(runner, name, case, config):
    hardware, model, execution, config = environment(config, case.get("fabric"))
    layout = layout_for(case["kvp"], case.get("pp", PP))
    counts = case.get("layer_counts", LAYER_COUNTS)
    parameters = dict(
        case=name,
        config=config,
        execution=execution.report(),
        layout=asdict(layout),
        layer_counts=counts,
        launch_us=case.get("launch_us", 0),
        kv_settings=kv_settings(case),
    )

    def execute(directory):
        result = screen(
            hardware,
            model,
            layout,
            launch_us=parameters["launch_us"],
            layer_counts=counts,
            token_units=config.get("token_units"),
            kv_settings=parameters["kv_settings"],
        )
        if result["status"] != "ok":
            raise ConfigurationError(result["reason"])
        return dict(
            parameters=parameters, result=result, source_sha256=source_identity()
        )

    receipt = runner.run(parameters, execute)
    tps = [row["generation_tps"] for row in receipt.result["result"]["workloads"]]
    print(json.dumps(dict(case=name, reused=receipt.reused, tps=tps)), flush=True)
    return receipt.result


def minimum_stages(config, tp_values=(8, 16, 32), kvp_values=(1, 2, 4, 8)):
    """Smallest memory-feasible PP for each (TP, KVP) under capacity partitions."""
    hardware, model, _, config = environment(config)
    rows = []
    for tp in tp_values:
        for kvp in (k for k in kvp_values if tp % k == 0):
            cache, partition, pp = {}, None, None
            settings = kv_settings(dict(kvp=kvp))
            for stages in range(1, min(model.num_layers, 256 // tp) + 1):
                try:
                    partition = capacity_partition(
                        hardware,
                        model,
                        layout_for(kvp, stages, tp),
                        MAXIMUM_CONTEXT,
                        cache,
                        kv_settings=settings,
                    )
                except ConfigurationError:
                    continue
                pp = stages
                break
            row = dict(
                tp=tp,
                kvp=kvp,
                minimum_pp=pp,
                minimum_lp30=pp and tp * pp,
                partition=partition,
            )
            rows.append(row)
            print(json.dumps(dict(tp=tp, kvp=kvp, minimum_pp=pp)), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-context-parallel"
    )
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=None)
    parser.add_argument(
        "--minimum",
        action="store_true",
        help="search memory-feasible PP per TP/KVP and screen the smallest TP16 one",
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    config = load(args.config)
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
    )
    names = args.cases or sorted(CASES)
    results = {name: run_case(runner, name, CASES[name], config) for name in names}
    minimum = None
    if args.minimum:
        minimum = minimum_stages(config)
        # Screen the smallest TP16 KV-parallel inventory below the baseline.
        row = min(
            (r for r in minimum if r["tp"] == TP and r["kvp"] > 1 and r["minimum_pp"]),
            key=lambda r: (r["minimum_pp"], -r["kvp"]),
        )
        if row["minimum_pp"] < PP:
            name = f"kvp{row['kvp']}-pp{row['minimum_pp']}"
            case = dict(
                kvp=row["kvp"],
                pp=row["minimum_pp"],
                layer_counts=list(row["partition"]["layer_counts"]),
            )
            results[name] = run_case(runner, name, case, config)
    if args.summary:
        write_json(
            args.summary,
            dict(
                source_sha256=source_identity(),
                cases={
                    name: dict(
                        parameters=value["parameters"],
                        tps=[
                            row["generation_tps"]
                            for row in value["result"]["workloads"]
                        ],
                        memory=value["result"]["memory"],
                    )
                    for name, value in results.items()
                },
                minimum_stages=minimum,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
