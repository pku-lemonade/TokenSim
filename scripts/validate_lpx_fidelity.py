"""Validate workspace admission, complete prompt prefill and native decode clocks."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
from time import perf_counter

import simpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TokenSim.af.serving.service import AFService
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.experiments import ExperimentRunner
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.fidelity import load, with_fidelity
from scripts.lpx.native_search import WORKLOADS, build_runtime, memory_report, screen
from scripts.lpx.partition import capacity_partition
from scripts.lpx.replay_profiles import source_identity
from scripts.search_lpx_native import write_json


def complete_prefill(runtime, inputs, outputs, chunk):
    env = simpy.Environment()
    service = AFService(env, runtime, trace=False, retain_trace=False)
    session = service.admit(0, 0, maximum_context=inputs + outputs, current_context=0)
    chunks = 0

    def run():
        nonlocal chunks
        while session.cached_context < inputs:
            query = min(chunk, inputs - session.cached_context)
            yield env.process(
                service.forward(
                    [session],
                    phase="prefill",
                    query_tokens=query,
                    emit_logits=session.cached_context + query == inputs,
                )
            )
            chunks += 1
            service.completed_passes.clear()

    started = perf_counter()
    env.run(until=env.process(run()))
    if session.cached_context != inputs:
        raise SimulationStateError(
            "complete prompt prefill did not reach target context"
        )
    peak = memory_report(runtime, inputs + outputs)
    service.release(session)
    if (
        service.sessions
        or service.memory.leases
        or any(service.memory.reserved.values())
    ):
        raise SimulationStateError("prefill validation leaked a KV lease")
    return dict(
        input_tokens=inputs,
        reserved_output_tokens=outputs,
        prefill_chunk=chunk,
        executed_chunks=chunks,
        final_context=inputs,
        simulated_prefill_seconds=env.now,
        host_seconds=perf_counter() - started,
        memory=peak,
        scope="Complete aggregate native prefill with workspace bounds; no real model numerics or endpoint TTFT calibration.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/lpx/fidelity-study.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-fidelity-validation"
    )
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--pp", type=int, default=8)
    parser.add_argument("--kvp", type=int, default=1, help="KV-parallel size")
    parser.add_argument(
        "--kv-parallel-layers",
        choices=["full_attention", "all"],
        default="full_attention",
    )
    parser.add_argument("--kv-append-block", type=int, default=16)
    args = parser.parse_args(argv)
    if args.kvp < 1 or args.tp % args.kvp:
        parser.error("--kvp must be a positive divisor of --tp")
    kv_settings = (
        dict(
            kv_parallel_layers=args.kv_parallel_layers,
            kv_append_block=args.kv_append_block,
        )
        if args.kvp > 1
        else None
    )
    config = load(args.config)
    if "memory" not in config:
        parser.error(
            "complete-prefill validation requires a memory configuration section"
        )
    execution = ExecutionAssumptions(
        mode="compiled", instruction_floor_us=0, c2c_hop_us=0.35
    )
    hardware = with_fidelity(
        with_execution(HardwareContext.load(ROOT / "data"), execution), config
    )
    model = replace(
        hardware.models.resolve("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
    )
    layout = Layout(
        a_device="groq_lp30",
        f_device="groq_lp30",
        a_tp=args.tp // args.kvp,
        a_cp=args.kvp,
        f_tp=args.tp,
        a_pp=args.pp,
        f_pp=args.pp,
        f_ep=1,
    )
    partition = capacity_partition(
        hardware,
        model,
        layout,
        max(i + o for i, o, _ in WORKLOADS),
        {},
        kv_settings=kv_settings,
    )
    parameters = dict(
        config=config,
        execution=execution.report(),
        model=model.to_dict(),
        layout=asdict(layout),
        partition=partition,
        **({"kv_settings": kv_settings} if kv_settings else {}),
    )
    runner = ExperimentRunner(
        args.output / "points",
        inputs=(ROOT / "data",),
        code_paths=(ROOT / "TokenSim", ROOT / "scripts/lpx", Path(__file__)),
    )

    def execute(directory):
        runtime = build_runtime(
            hardware,
            model,
            layout,
            layer_counts=partition["layer_counts"],
            kv_settings=kv_settings,
        )
        prefill = complete_prefill(
            runtime, 100000, 2000, config["memory"]["prefill_chunk"]
        )
        result = screen(
            hardware,
            model,
            layout,
            launch_us=0,
            verify=True,
            requests=50,
            warmup=5,
            layer_counts=partition["layer_counts"],
            token_units=config.get("token_units"),
            kv_settings=kv_settings,
        )
        if result["status"] != "ok":
            raise ConfigurationError(result["reason"])
        return dict(
            parameters=parameters,
            prefill=prefill,
            decode=result,
            source_sha256=source_identity(),
        )

    receipt = runner.run(parameters, execute)
    write_json(args.output / "results.json", receipt.result)
    print(
        dict(
            reused=receipt.reused,
            tps=[w["generation_tps"] for w in receipt.result["decode"]["workloads"]],
            prefill=receipt.result["prefill"]["executed_chunks"],
        )
    )
    return receipt.result


if __name__ == "__main__":
    main()
