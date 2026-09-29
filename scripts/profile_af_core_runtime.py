"""可复现的 AF 迁移基线；从仓库根目录以 python -m scripts.profile_af_core_runtime 运行。"""

import argparse
import cProfile
import json
import pstats
from pathlib import Path
from time import perf_counter

from scripts.deepseek_v3_runtime import configuration, simulate
from tests.af_runtime_oracle import observe_services
from tests.test_af_engine_integration import (
    _build_engine,
    _disabled_cluster,
    _run_request,
)
from TokenSim.llm.llm_request import reset_g_time
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.af.factory import build_model_runtime
import simpy


def profile_case(run):
    profile = cProfile.Profile()
    start = perf_counter()
    with observe_services() as services:
        profile.enable()
        result = run()
        profile.disable()
    elapsed = perf_counter() - start
    costs = {name: 0.0 for name in ("planning", "lookup", "simpy", "trace", "other")}
    for (filename, _, _), (_, _, self_time, _, _) in pstats.Stats(
        profile
    ).stats.items():
        normalized = filename.replace("\\", "/")
        category = (
            "planning"
            if any(x in normalized for x in ("/af/planning.py", "/af/templates.py"))
            else (
                "lookup"
                if any(
                    x in normalized
                    for x in (
                        "/operator_data/",
                        "/operator_provider.py",
                        "/latency/operator_table.py",
                    )
                )
                else (
                    "simpy"
                    if "/simpy/" in normalized
                    else "trace" if "/af/trace.py" in normalized else "other"
                )
            )
        )
        costs[category] += self_time
    return dict(
        result=result,
        wall_seconds=elapsed,
        profile_self_seconds=costs,
        service_count=len(services),
        communication_bytes=sum(
            row["estimate"]["payload_bytes"] or 0 for row in services
        ),
        services=services,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    config = configuration(full_model=args.full_model, af_batch_size=1)
    config.cluster.af.service_estimate_cache = not args.no_cache
    config.cluster.af.dag_template_cache = not args.no_cache
    cases = {}
    for trace in (True, False):

        def run():
            executor, rows, compute, transfer = simulate(
                config, 2, 8, steps=2, trace=trace, fallback="analytical_only"
            )
            return dict(
                iterations=rows,
                operator_stats=compute.stats_dict(),
                optimization_stats=dict(
                    template_requests=executor.plan_builder.template_cache.requests,
                    template_hits=executor.plan_builder.template_cache.hits,
                    transfer_requests=transfer.estimate_requests,
                    transfer_hits=transfer.estimate_cache_hits,
                ),
            )

        cases[f"script_trace_{trace}"] = profile_case(run)

    def disabled():
        reset_g_time()
        env, engine, _ = _build_engine(_disabled_cluster())
        request = _run_request(env, engine)
        return dict(
            completed=request.is_done,
            completion_time=request.total_time,
            ttft=request.prefill_latency,
            decode_time_sum=request.decode_time_sum,
        )

    cases["af_disabled"] = profile_case(disabled)

    def standard():
        reset_g_time()
        hardware = config.hardware
        doc = json.loads(Path("data/clusters/deepseek_v3/1a32f.json").read_text())
        doc["af"]["service_estimate_cache"] = not args.no_cache
        doc["af"]["dag_template_cache"] = not args.no_cache
        runtime = build_model_runtime(
            doc, hardware=hardware, model=hardware.model(config.model.model_id)
        )
        env = simpy.Environment()
        engine = LLMEngine(
            env=env,
            block_size=16,
            batching="paged-attn",
            kv_transfer_config=KVTransferConfig.default(),
            psla_config=PSLAConfig.from_file("data/psla/deepseek-v3.json"),
            cluster_config=runtime.cluster,
            hardware=hardware,
            prefill_worker_pool_type="round_robin",
            decode_worker_pool_type="round_robin",
            max_parallem_sum=8,
            latency_fallback="analytical_only",
        )
        request = _run_request(env, engine, decode_len=2)
        return dict(
            completed=request.is_done,
            completion_time=request.total_time,
            ttft=request.prefill_latency,
            decode_time_sum=request.decode_time_sum,
            operator_stats=engine.af_compute_provider.stats_dict(),
        )

    cases["standard_v3"] = profile_case(standard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                workload=dict(
                    batch=2,
                    context=8,
                    decode_steps=2,
                    layers=config.model.num_layers,
                    fallback="analytical_only",
                    seed=42,
                ),
                note="墙钟包含 cProfile 和差分观察器开销；只用于相同仪器下比较。AF-disabled 是非 MLA 兼容基线。",
                cases=cases,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
