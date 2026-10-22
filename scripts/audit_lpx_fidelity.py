"""Run incremental LP30 fidelity ablations with checked, resumable results."""

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import gc
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.fidelity import load, with_fidelity
from scripts.lpx.native_search import WORKLOADS, screen
from scripts.lpx.partition import capacity_partition
from scripts.lpx.replay_profiles import source_identity
from scripts.lpx.token_units import TokenUnits
from scripts.search_lpx_native import write_json


def variants(config):
    common = {"compute": config["compute"]}
    cases = [("legacy-112", 16, 7, common), ("legacy-136", 8, 17, common)]
    for part in ("operator", "fabric", "memory"):
        cases.append((part + "-112", 16, 7, {**common, part: config[part]}))
    cases.extend(
        (name, tp, pp, config)
        for name, tp, pp in (
            ("integrated-112", 16, 7),
            ("integrated-128", 16, 8),
            ("integrated-136", 8, 17),
        )
    )
    for name, section, changes in (
        (
            "vector-fast-136",
            "operator",
            dict(vector_ops_per_s=50e12, exp_ops_per_s=2.5e12),
        ),
        (
            "vector-slow-136",
            "operator",
            dict(vector_ops_per_s=1e12, exp_ops_per_s=1e11),
        ),
        ("serialized-136", "operator", dict(schedule="serialized")),
        ("one-lane-136", "fabric", dict(lanes_per_peer=1, p2p_lanes=1)),
    ):
        altered = deepcopy(config)
        altered[section].update(changes)
        cases.append((name, 8, 17, altered))
    for ratio in (0.8, 1.2):
        altered = deepcopy(config)
        altered["token_units"].update(
            native_per_aa_input=ratio,
            native_per_aa_output=ratio,
            source="Unmeasured token-ratio sensitivity assumption.",
            measured=False,
        )
        altered["memory"]["maximum_context"] = int(102000 * max(1, ratio))
        cases.append((f"token-ratio-{ratio}-128", 16, 8, altered))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-fidelity-study"
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args(argv)
    config = load(args.config)
    hardware = HardwareContext.load(ROOT / "data")
    model = replace(
        hardware.models.resolve("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
    )
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
    )
    started, rows, receipts = perf_counter(), [], []
    for name, tp, pp, fidelity in variants(config):
        if args.cases and name not in args.cases:
            continue
        execution = ExecutionAssumptions(
            mode="compiled",
            instruction_floor_us=0.0 if "operator" in fidelity else 0.05,
            c2c_hop_us=0.35,
        )
        layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=tp,
            f_tp=tp,
            a_pp=pp,
            f_pp=pp,
            f_ep=1,
        )
        parameters = dict(
            name=name,
            model=model.to_dict(),
            layout=asdict(layout),
            execution=execution.report(),
            fidelity=fidelity,
            verify=args.verify,
        )

        def execute(directory):
            from TokenSim.errors import ConfigurationError

            catalog = with_fidelity(with_execution(hardware, execution), fidelity)
            try:
                partition = capacity_partition(
                    catalog,
                    model,
                    layout,
                    max(
                        sum(
                            TokenUnits(
                                **fidelity.get("token_units", {})
                            ).native_lengths(i, o)
                        )
                        for i, o, _ in WORKLOADS
                    ),
                    {},
                )
            except ConfigurationError as exc:
                return dict(
                    status="infeasible",
                    reason=str(exc),
                    lp30=layout.a_ranks,
                    layout=asdict(layout),
                )
            result = screen(
                catalog,
                model,
                layout,
                launch_us=0,
                layer_counts=partition["layer_counts"],
                verify=args.verify,
                token_units=fidelity.get("token_units"),
            )
            result["partition_evidence"] = partition
            return result

        receipt = runner.run(parameters, execute)
        rows.append(dict(name=name, parameters=parameters, result=receipt.result))
        receipts.append(dict(name=name, key=receipt.key, reused=receipt.reused))
        write_json(args.output / "progress.json", dict(rows=rows, receipts=receipts))
        print(
            dict(
                name=name,
                status=receipt.result["status"],
                tps=[
                    round(w["generation_tps"], 3)
                    for w in receipt.result.get("workloads", [])
                ],
            ),
            flush=True,
        )
        gc.collect()
    result = dict(
        source_sha256=source_identity(),
        config=config,
        rows=rows,
        receipts=receipts,
        host_seconds=perf_counter() - started,
        method="Native decode graph refinement and adaptive integer-context curves; deterministic equal-length serial clients.",
    )
    write_json(args.output / "results.json", result)
    print(dict(host_seconds=result["host_seconds"]), flush=True)
    return result


if __name__ == "__main__":
    main()
