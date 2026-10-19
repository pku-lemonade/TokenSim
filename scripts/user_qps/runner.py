"""Run real AF iterations and save portable evidence for each offered QPS."""

from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
from types import SimpleNamespace

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.errors import SimulationStateError
from scripts.user_qps.metrics import validate_and_summarize
from scripts.user_qps.scheduler import schedule, workload

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ("arrivals", "requests", "tokens", "batches", "queue")


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def experiment(args, runtime):
    files = {
        *ROOT.joinpath("TokenSim").rglob("*.py"),
        *ROOT.joinpath("scripts").rglob("*.py"),
        *ROOT.joinpath("data").rglob("*.yaml"),
        *ROOT.joinpath("data").rglob("*.json"),
    }
    hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if not {"validation", "results", "__pycache__"}.intersection(path.parts)
    }
    return dict(
        schema="tokensim.user_qps.v1",
        arguments={**vars(args), "output": str(args.output)},
        model=runtime.model.to_dict(),
        python=sys.version,
        platform=platform.platform(),
        layers=61,
        timing="analytical_only",
        phase="decode_only",
        hypothetical_f_memory_gib=1536,
        assumptions=[
            "F memory is virtual capacity; compute and HBM bandwidth use A100 specifications.",
            "Each request starts with an independent resident KV prefix; prefill is excluded.",
            "Finite arrivals from empty queues followed by a complete drain; no steady-state claim.",
            "Current repository execution and analytical estimates can differ from archived revisions.",
        ],
        routing_step="Global iteration; context and output token index remain request-local",
        metric_definitions=dict(
            ttft="arrival to first output token, including queueing, excluding prefill",
            tbt="adjacent output-token intervals within each request",
            latency="arrival to last output token",
            throughput="all output tokens / simulated time from zero to last completion",
            percentile="nearest rank, ceil(p*N)",
            wall_seconds="host execution time, separate from simulated latency",
        ),
        source_sha256=hashes,
    )


def prepare(args, runtime, qps):
    output = args.output / f"qps-{qps:g}"
    output.mkdir()
    dump(output / "cluster.json", runtime.doc)
    dump(output / "model.json", runtime.model.to_dict())
    dump(output / "runtime-plan.json", runtime.plan.summary())
    dump(output / "effective-af.json", asdict(runtime.cluster.af))
    dump(
        output / "hardware.json",
        dict(
            devices={
                worker.hardware: runtime.hardware.device(worker.hardware).to_dict()
                for worker in runtime.workers
            },
            link=vars(runtime.hardware.link("nvlink3")),
        ),
    )
    arrivals = workload(qps, args.requests, args.users, args.seed, args.unique_users)
    dump(output / "arrivals.json", arrivals)
    return output, arrivals


def run_case(args, runtime, qps):
    output, arrivals = prepare(args, runtime, qps)
    env = simpy.Environment()
    compute, transfer = build_operator_table_af_providers(
        hardware=runtime.hardware,
        model=runtime.model,
        cluster_config=runtime.cluster,
        worker_configs=runtime.workers,
        placement=runtime.placement,
        dtype="bf16",
        fallback="analytical_only",
    )
    dump(output / "communication.json", transfer.describe())
    if args.dry_run:
        dump(output / "progress.json", dict(status="planned", requests=len(arrivals)))
        print(f"Validated configuration and arrivals: {output}", flush=True)
        return None
    executor = ModelAFExecutor(
        env,
        runtime.cluster,
        runtime.workers,
        compute,
        transfer,
        runtime.model.model_id,
        "bf16",
        61,
        runtime.model.hidden_size,
        model=runtime.model,
        moe_config=runtime.model.moe,
        random_seed=args.seed,
        runtime_plan=runtime.plan,
        trace_recorder=AFTraceRecorder(False),
    )
    executor.trace.resource_capacities = executor.resources.resource_capacities()
    owner = next(iter(runtime.plan.request_owners))
    source_worker = next(
        index
        for index, worker in enumerate(runtime.workers)
        if worker.physical_worker_ref.identity == owner.physical_worker
    )
    records = dict(arrivals=arrivals, requests=[], tokens=[], batches=[], queue=[])
    sampled = set()
    started = time.monotonic()

    def progress(iterations, active, pending):
        row = dict(
            status="running",
            qps=qps,
            iterations=iterations,
            completed_requests=sum(
                r["status"] == "complete" for r in records["requests"]
            ),
            total_requests=args.requests,
            generated_tokens=len(records["tokens"]),
            active_users=active,
            pending=pending,
            simulated_seconds=env.now,
            wall_seconds=time.monotonic() - started,
        )
        dump(output / "progress.json", row)
        print(json.dumps(row), flush=True)

    try:
        progress(0, 0, 0)
        with gzip.open(
            output / "sample-operators.trace.json.gz",
            "wt",
            encoding="utf-8",
            compresslevel=1,
        ) as stream:
            stream.write(
                '{"metadata":{"scope":"first occurrence of each actual batch size","timing":"SIMULATED"},"traceEvents":['
            )
            event_count = 0

            def serve(requests, iteration_id):
                nonlocal event_count
                executor.trace.enabled = len(requests) not in sampled
                routed = [
                    SimpleNamespace(**{**vars(r), "generation_idx": iteration_id})
                    for r in requests
                ]
                iteration = executor.submit(source_worker, routed)
                yield env.process(executor.run_iteration(iteration))
                if executor.trace.enabled:
                    events = executor.trace.events_for(iteration.iteration_id)
                    layers = {event.layer_id for event in events if event.layer_id >= 0}
                    if layers != set(range(61)):
                        raise SimulationStateError(
                            "sampled trace does not cover all 61 layers"
                        )
                    for event in events:
                        if event_count:
                            stream.write(",")
                        json.dump(
                            dict(
                                name=event.operation or event.stage or event.event_type,
                                cat=event.event_type,
                                ph="X",
                                pid=1,
                                tid=event.resource_id,
                                ts=event.start_at * 1e6,
                                dur=event.service_duration * 1e6,
                                args=event.as_dict(),
                            ),
                            stream,
                        )
                        event_count += 1
                    sampled.add(len(requests))
                    executor.trace.clear()
                    stream.flush()
                return {
                    request_id: iteration.block_completed_at[block.block_id]
                    for block in iteration.blocks
                    for request_id in block.request_ids
                }

            schedule(
                env,
                arrivals,
                users=args.users,
                max_batch=args.max_batch,
                output_tokens=args.output_tokens,
                context=args.context,
                serve=serve,
                request_rows=records["requests"],
                token_rows=records["tokens"],
                batch_rows=records["batches"],
                queue_rows=records["queue"],
                busy_policy=args.busy_policy,
                progress=progress,
            )
            env.run()
            stream.write("]}")
        for name in RECORDS:
            dump(output / f"{name}.json", records[name])
        # Audit the files third parties receive, including every emitted token.
        persisted = [
            json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
            for name in RECORDS
        ]
        summary = validate_and_summarize(args, qps, *persisted)
        domains = executor.resources.domain_scheduler
        if domains.owners or domains.waiters:
            raise SimulationStateError("communication resources did not drain")
        dump(
            output / "validation.json",
            dict(
                status="passed",
                token_accounting="passed",
                user_serialization="passed",
                final_queues_empty=True,
                communication_queues_empty=True,
                sampled_batch_sizes=sorted(sampled),
                sampled_layers=61,
                profile_events=event_count,
            ),
        )
        summary["wall_seconds"] = time.monotonic() - started
        summary["operator_stats"] = compute.stats_dict()
        dump(output / "summary.json", summary)
        dump(output / "progress.json", dict(status="complete", **summary))
        print(json.dumps(summary), flush=True)
        return summary
    except BaseException as exc:
        for name in RECORDS:
            dump(output / f"{name}.json", records[name])
        dump(
            output / "failure.json",
            dict(status="failed", error=str(exc), traceback=traceback.format_exc()),
        )
        dump(output / "progress.json", dict(status="failed", error=str(exc)))
        raise
