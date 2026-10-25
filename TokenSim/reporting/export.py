"""Write result JSON, missing operator shapes and failure diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
from pathlib import Path
from dataclasses import asdict
from typing import Any
from TokenSim.results import LLMResult, MetricData
from TokenSim.config.config import ClusterConfig
from TokenSim.llm.llm_request import LLMTime, Request
from .metrics import RunReport, get_agentx_metrics, slo_report
from .af import get_af_stats
from .stats import (
    get_prefix_reuse_stats,
    get_connector_stats,
    get_parallel_stats,
    get_cache_capacity_stats,
    get_latency_stats,
    get_moe_stats,
    get_mooncake_stats,
)

if TYPE_CHECKING:
    from TokenSim.llm.llm_engine import LLMEngine


def export_result(
    run: RunReport,
    g_time: LLMTime,
    engine: LLMEngine,
    cluster: ClusterConfig,
    request_count: int,
    prefill_lens: list[int],
    decode_lens: list[int],
    requests: list[Request],
    notdone: list[int],
    duration: float,
    simulator_wall_time: float = 0,
):
    if run.agentx_concurrency is not None:
        request_time = MetricData.from_list([req.total_time for req in requests])
        first_token = [req.prefill_latency for req in requests]
        decode_token = [
            req.decode_time_sum / max(1, req.generation_idx - 1) for req in requests
        ]
    else:
        request_time = MetricData.from_list(g_time.request_time)
        first_token, decode_token = g_time.prefill_time, g_time.decode_time
    prefill_time = MetricData.from_list(first_token)
    decode_time = MetricData.from_list(decode_token)
    prefix_reuse_stats = get_prefix_reuse_stats(requests)
    connector_stats = get_connector_stats(engine)
    parallel_stats = get_parallel_stats(engine)
    cache_capacity_stats = get_cache_capacity_stats(engine)
    moe_stats = get_moe_stats(engine)
    mooncake_stats = get_mooncake_stats(engine)
    latency_stats = get_latency_stats(engine)

    result = LLMResult(
        qps=run.qps,
        cluster=cluster,
        request_time=request_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        request_count=request_count,
        prefill_lens=prefill_lens,
        decode_lens=decode_lens,
        batching=run.batching,
        duration=duration,
        simulator_wall_time=simulator_wall_time,
        simulated_time=duration,
        output_qps=sum(r.is_done for r in requests) / duration if duration else 0,
        output_token_ps=(
            sum(r.generation_idx for r in requests) / duration if duration else 0
        ),
        notdone=notdone,
        termination_reason=run.termination_reason,
        failure=run.failure,
        preemption_count=sum(worker.preempted_cnt for worker in engine.workers),
        recomputation_count=sum(req.recomputation_count for req in requests),
        recomputed_tokens=sum(req.recomputed_tokens_total for req in requests),
        recompute_service_time=sum(req.recompute_service_time for req in requests),
        prefill_chunk_count=sum(req.prefill_chunks for req in requests),
        max_num_batched_tokens=getattr(engine, "max_num_batched_tokens", None),
        router_stats=(
            engine.router.describe() if getattr(engine, "router", None) else None
        ),
        dp_step_stats=(
            engine.dp_coordinator.describe()
            if getattr(engine, "dp_coordinator", None)
            else None
        ),
        scheduler_stats=[
            {
                "worker_id": w.id,
                "long_prefill_token_threshold": getattr(
                    getattr(w, "scheduler", None), "long_prefill_token_threshold", None
                ),
                "prefill_schedule_interval": getattr(
                    w, "prefill_schedule_interval", None
                ),
                "counts": dict(getattr(getattr(w, "scheduler", None), "stats", {})),
            }
            for w in engine.workers
        ],
        **prefix_reuse_stats,
        **connector_stats,
        **parallel_stats,
        **cache_capacity_stats,
        **moe_stats,
        **mooncake_stats,
        **latency_stats,
        **get_af_stats(engine),
        agentx_metrics=get_agentx_metrics(
            run,
            requests,
            duration,
            num_gpus=len(engine.workers),
        ),
        slo=slo_report(
            getattr(engine, "workload_config", None),
            first_token,
            decode_token,
            duration,
        ),
    )

    results_path = Path(run.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    result_dict = {
        key: value for key, value in asdict(result).items() if value is not None
    }
    result_dict["cluster"] = asdict(result.cluster)
    result_dict["request_time"] = asdict(result.request_time)
    result_dict["prefill_time"] = asdict(result.prefill_time)
    result_dict["decode_time"] = asdict(result.decode_time)

    if getattr(engine, "af_executor", None) is None:
        for name in (
            "af_enabled",
            "af_iteration_count",
            "af_trace",
            "af_runtime_state",
            "af_latency_providers",
            "af_operator_stats",
        ):
            result_dict.pop(name, None)
    if engine.trace_session is not None:
        result_dict["trace"] = engine.trace_session.export(
            results_path,
            termination=run.termination_reason,
            result_name=run.result_name,
        )
    result_file = results_path / run.result_name
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)
    if latency_stats.get("operator_missing_shape_groups"):
        # Shapes the tables could not answer, grouped by table and discrete key
        # with the requested axis ranges: the collection checklist for the next
        # profiling run on the target device (compute and communication alike).
        missing_file = results_path / f"missing_shapes_{run.qps}.json"
        with open(missing_file, "w") as f:
            json.dump(_missing_shape_report(engine), f, indent=2)
    return result_dict


def _missing_shape_report(engine: LLMEngine) -> dict[str, Any]:
    from TokenSim.operator_data.coverage import MissingShapeReport

    report = MissingShapeReport()
    for worker in getattr(engine, "workers", []):
        worker_report = getattr(worker, "missing_report", None)
        if worker_report is not None:
            report = report.merge(worker_report)
    return report.to_dict()


def write_failure(run, env, engine, unfinished, error):
    path = Path(run.results_path)
    path.mkdir(parents=True, exist_ok=True)
    diagnostic = {
        "termination_reason": run.termination_reason,
        "simulation_time": env.now,
        "failure": str(error) if error else None,
        "unfinished_request_ids": unfinished,
        "unfinished_af_iterations": (
            engine.af_executor.unfinished_snapshot()
            if getattr(engine, "af_executor", None) is not None
            else []
        ),
        "workers": [
            {
                "worker_id": w.id,
                "running": [
                    r.id for r in getattr(getattr(w, "scheduler", None), "running", [])
                ],
                "waiting": [
                    r.id for r in getattr(getattr(w, "scheduler", None), "waiting", [])
                ],
            }
            for w in engine.workers
        ],
    }
    (path / "failure.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    if error:
        with (path / "failed.txt").open("a") as stream:
            stream.write(str(error) + "\n")
