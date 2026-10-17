"""Exercise the production AF DAG on one resident 2T-proxy layer."""

from dataclasses import replace

from TokenSim.af.factory import build_model_runtime, simulate_decode
from TokenSim.errors import ConfigurationError


def cluster_doc(batch):
    groups = [
        dict(
            name="a",
            capabilities=["af_attention"],
            hardware="rubin",
            num_workers=8,
            network="af-net",
        ),
        dict(
            name="f",
            capabilities=["af_ffn"],
            hardware="groq_lp30",
            num_workers=256,
            network="af-net",
        ),
    ]
    domains = []
    for name, tp, dp, capability in (
        ("a", 8, 1, "af_attention"),
        ("f", 32, 8, "af_ffn"),
    ):
        instances = []
        for replica in range(dp):
            mappings = [
                dict(
                    domain_rank=replica * tp + rank,
                    tp_rank=rank,
                    pp_rank=0,
                    dp_rank=replica,
                    ep_rank=replica * tp + rank if name == "f" else 0,
                    physical_worker=dict(
                        group_name=name, worker_index=replica * tp + rank
                    ),
                )
                for rank in range(tp)
            ]
            instances.append(dict(name=f"{name}{replica}", rank_mapping=mappings))
        domains.append(
            dict(
                name=name,
                capability=capability,
                phases=["prefill", "decode"],
                worker_group=name,
                parallel=dict(
                    tensor_parallel_size=tp,
                    data_parallel_size=dp,
                    expert_parallel_size=tp * dp if name == "f" else 1,
                ),
                logical_instances=instances,
                communication_plan=(
                    [
                        dict(
                            operation=operation,
                            after_stage=stage,
                            participants=list(range(tp)),
                            network="a-net",
                            resource_group="a-collective",
                            resource_lane="collective",
                        )
                        for operation, stage in (
                            ("all_reduce", "a_compute"),
                            ("all_gather", "residual"),
                        )
                    ]
                    if name == "a"
                    else []
                ),
            )
        )
    return dict(
        num_workers=264,
        topology="rubin_nvl72",
        networks={"af-net": "ethernet1600Gb", "a-net": "nvlink6"},
        worker_groups=groups,
        af=dict(
            enabled=True,
            phases=["prefill", "decode"],
            af_block_size=batch,
            latency_provider="operator_table",
            trace_enabled=True,
            transfer_topology_mode="flat_link",
            router_location="a",
            activation_dtype="bf16",
            kv_cache_dtype="bf16",
            communication_domains=dict(
                node_size=8, node_count=33, placement_strategy="dedicated"
            ),
            execution_domains=domains,
            communication_fabrics=[
                dict(
                    resource_group="af-fabric",
                    network="af-net",
                    duplex="full",
                    lanes=[
                        dict(name="a_to_f", capacity=1),
                        dict(name="f_to_a", capacity=1),
                    ],
                ),
                dict(
                    resource_group="a-collective",
                    network="a-net",
                    duplex="half",
                    lanes=[dict(name="collective", capacity=1)],
                ),
            ],
            links=[
                dict(
                    name="a-f",
                    phases=["prefill", "decode"],
                    attention_group="a",
                    ffn_group="f",
                    attention_domain="a",
                    ffn_domain="f",
                    network="af-net",
                    resource_group="af-fabric",
                    duplex="full",
                    a_to_f_capacity=1,
                    f_to_a_capacity=1,
                )
            ],
        ),
    )


def run_audit(hardware, target, *, batches=(1, 4, 16), context=404000):
    layer = replace(
        target,
        num_layers=1,
        attention_windows=(0,),
        moe=replace(target.moe, num_moe_layers=1),
    )
    rows = []
    for batch in batches:
        runtime = build_model_runtime(
            cluster_doc(batch), hardware=hardware, model=layer
        )
        executor, measurements, compute, transfer = simulate_decode(
            runtime, batch, context, trace=True, fallback="analytical_only"
        )
        rows.append(
            {
                "batch": batch,
                "measurements": measurements,
                "runtime_plan": runtime.plan.summary(),
                "operator_stats": compute.stats_dict(),
                "transfer": transfer.describe(),
                "events": [event.as_dict() for event in executor.trace.events],
            }
        )
    try:
        build_model_runtime(cluster_doc(1), hardware=hardware, model=target)
    except ConfigurationError as exc:
        rejected = str(exc)
    else:
        raise AssertionError("unresident whole-model LP30 placement must be rejected")
    return {
        "status": "production_af_one_layer_audit",
        "model": target.model_id,
        "layers_executed": 1,
        "layers_in_target": target.num_layers,
        "whole_model_negative_control": rejected,
        "note": "One resident layer on 8 Rubin + 256 LP30 ranks; no full-model throughput extrapolation.",
        "results": rows,
    }
