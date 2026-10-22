"""Validate workspace budgets and execute a complete serial prompt on LP30."""

import argparse
from dataclasses import replace
import hashlib
import json
from math import isclose
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.execution import ExecutionAssumptions, with_execution
from scripts.lpx.native_config import configuration
from scripts.lpx.native_decode import SerialDecodeDriver, curve_for, interval_summary
from scripts.lpx.native_search import memory_report, streaming_settings
from scripts.lpx.partition import native_footprint, partition_solutions
from scripts.lpx.replay_profiles import source_identity


def capacity_matrix(hardware, model):
    rows = []
    for tp in (8, 16, 32):
        pp = min(60, 256 // tp)
        layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=tp,
            f_tp=tp,
            a_pp=pp,
            f_pp=pp,
            f_ep=1,
        )
        base = configuration(
            hardware, model, layout, namespace="workspace", colocated_dense_reuse=True
        )
        footprint = native_footprint(base, 102000)
        for chunk in (1, 4, 8, 16, 32, 64, 128, 256):
            candidate = replace(
                footprint, buffers=tuple(n * chunk for n in footprint.buffers)
            )
            solutions = partition_solutions(candidate, pp)
            row = dict(
                tp=tp, prefill_chunk=chunk, minimum_pp=min(solutions, default=None)
            )
            if solutions:
                stages = min(solutions)
                counts = solutions[stages]["layer_counts"]
                actual = configuration(
                    hardware,
                    model,
                    replace(layout, a_pp=stages, f_pp=stages),
                    namespace="workspace",
                    micro_batch=chunk,
                    colocated_dense_reuse=True,
                    pipeline_layer_counts=counts,
                )
                assert native_footprint(actual, 102000) == candidate
                report = memory_report(actual, 102000)
                row.update(chips=tp * stages, counts=counts, memory=report)
            rows.append(row)
    return rows


def complete_request(hardware, model, tp, pp, chunk, inputs, outputs, *, full):
    started = perf_counter()
    layout = Layout(
        a_device="groq_lp30",
        f_device="groq_lp30",
        a_tp=tp,
        f_tp=tp,
        a_pp=pp,
        f_pp=pp,
        f_ep=1,
    )
    seed_pp = min(model.num_layers, 256 // tp)
    seed_layout = replace(layout, a_pp=seed_pp, f_pp=seed_pp)
    seed = configuration(
        hardware,
        model,
        seed_layout,
        namespace="prefill",
        micro_batch=chunk,
        colocated_dense_reuse=True,
    )
    allocation = partition_solutions(native_footprint(seed, 102000), pp)[pp]
    runtime = configuration(
        hardware,
        model,
        layout,
        namespace="prefill",
        micro_batch=chunk,
        colocated_dense_reuse=True,
        pipeline_layer_counts=allocation["layer_counts"],
    )
    memory = memory_report(runtime, 102000)
    env = simpy.Environment()
    service = AFService(env, runtime, trace=False, retain_trace=False)
    session = service.admit(0, 0, maximum_context=inputs + outputs, current_context=0)

    def prefill():
        chunks = 0
        while session.cached_context < inputs:
            width = min(chunk, inputs - session.cached_context)
            final = session.cached_context + width == inputs
            yield env.process(
                service.forward(
                    [session], phase="prefill", query_tokens=width, emit_logits=final
                )
            )
            service.completed_passes.clear()
            chunks += 1
            if chunks % 1000 == 0:
                print(
                    json.dumps(
                        dict(
                            stage="prefill",
                            cached_context=session.cached_context,
                            host_seconds=perf_counter() - started,
                        )
                    ),
                    flush=True,
                )
        return chunks

    if full:
        task = env.process(prefill())
        env.run(until=task)
        chunks = task.value
    else:
        # Probe the most demanding context without claiming a full prompt run.
        service.memory.commit(session.request_id, inputs - chunk)
        session.cached_context = inputs - chunk
        env.run(
            until=env.process(
                service.forward(
                    [session], phase="prefill", query_tokens=chunk, emit_logits=True
                )
            )
        )
        service.completed_passes.clear()
        chunks = 1
    assert session.cached_context == inputs
    for rank, backend in service.prefill.backends.items():
        binding = runtime.plan.binding(*rank)
        expected = allocation["layer_counts"][binding.pp_rank]
        assert backend.layers_local == expected
        assert sum(c.layers_local for c in backend._layer_backends) == expected
    prefill_seconds = env.now
    session.generated = 1
    curve = curve_for(runtime, inputs, outputs, streaming_settings(layout), 1.0)
    expected = interval_summary(curve, inputs, outputs)
    driver = SerialDecodeDriver(curve)
    service.execution_driver = driver
    first = env.now

    def decode():
        for _ in range(outputs - 1):
            yield env.process(service.forward([session], phase="decode"))
            session.generated += 1
            service.completed_passes.clear()

    env.run(until=env.process(decode()))
    elapsed = env.now - first
    assert isclose(elapsed, expected["generation_seconds"], rel_tol=1e-9)
    assert session.cached_context == inputs + outputs - 1
    service.release(session)
    assert not service.sessions and not service.memory.leases
    assert not any(service.memory.reserved.values())
    return dict(
        tp=tp,
        pp=pp,
        chips=tp * pp,
        prefill_chunk=chunk,
        input_tokens=inputs,
        output_tokens=outputs,
        pipeline_layer_counts=allocation["layer_counts"],
        memory=memory,
        full_prefill_executed=full,
        executed_prefill_chunks=chunks,
        prefill_aggregate_model_seconds=prefill_seconds,
        executed_decode_steps=driver.steps,
        generation_tps=(outputs - 1) / elapsed,
        prefill_rank_backends=len(service.prefill.backends),
        host_seconds=perf_counter() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/lpx-execution-study"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assumptions = ExecutionAssumptions(
        mode="compiled", instruction_floor_us=0.05, c2c_hop_us=0.35, invocation_us=1
    )
    hardware = with_execution(HardwareContext.load(ROOT / "data"), assumptions)
    model = replace(
        hardware.model("gemma-4-31b-text"), dtype="fp8", kv_cache_dtype="fp8"
    )
    matrix = capacity_matrix(hardware, model)
    (out / "prefill-workspace-capacity.json").write_text(
        json.dumps(
            dict(
                source_sha256=source_identity(),
                validation_script_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                scope="Configured boundary buffers and the native allocator watermark; internal kernel workspace remains unverified.",
                rows=matrix,
            ),
            indent=2,
        )
        + "\n"
    )
    results = []
    for tp, pp, chunk, inputs, outputs, full in (
        (8, 11, 16, 1000, 1000, True),
        (16, 7, 4, 100000, 2000, False),
        (8, 11, 16, 100000, 2000, True),
    ):
        result = complete_request(
            hardware, model, tp, pp, chunk, inputs, outputs, full=full
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        (out / "prefill-validation.json").write_text(
            json.dumps(
                dict(
                    source_sha256=source_identity(),
                    validation_script_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    execution=assumptions.report(),
                    scope="Native aggregate chunked prefill, then sampled native-DAG serial decode; no inter-request contention.",
                    limitations=[
                        "Prefill timing does not use the decode vector-overlap refinement and is not a TTFT hardware prediction.",
                        "SRAM admission includes configured boundary buffers; additional internal kernel workspace remains unverified.",
                        "The 112-chip case probes the final prefill chunk and does not execute the complete prompt.",
                    ],
                    results=results,
                ),
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
