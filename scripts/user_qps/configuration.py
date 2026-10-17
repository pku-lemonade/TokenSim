"""Explicit hardware assumptions for the historical 32-user experiment."""

from copy import deepcopy

from TokenSim.af.factory import build_model_runtime
from TokenSim.errors import ConfigurationError
from scripts.deepseek_v3_runtime import configuration


def historical_runtime(args):
    base = configuration(
        full_model=True,
        a_tp=args.a_tp,
        f_tp=args.f_ep,
        af_batch_size=args.af_batch_size,
        device_id="a100_sxm_80g",
        network="nvlink3",
        hypothetical_f_memory_gib=1536,
    )
    document = deepcopy(base.doc)
    document["af"].update(
        transfer_topology_mode="physical",
        communication_resource_scope="fabric",
        communication_timing={"enabled": False},
        communication_domains=dict(
            node_size=8,
            node_count=5,
            placement_strategy="dedicated",
            local_bandwidth_bytes_per_s=300e9,
            inter_tx_bandwidth_bytes_per_s=50e9,
            inter_rx_bandwidth_bytes_per_s=50e9,
            local_latency_us=0,
            inter_latency_us=0,
        ),
        routed_expert_execution="grouped",
        transport_router_logits=False,
        router_location="a",
        execution_protocol="partial_return",
        compute_execution_mode="eager",
    )
    runtime = build_model_runtime(document, hardware=base.hardware, model=base.model)
    if runtime.plan.total_layers != 61 or runtime.model.dtype != "bf16":
        raise ConfigurationError("the user experiment requires all 61 BF16 V3 layers")
    peak_context = args.context + args.output_tokens
    if peak_context > runtime.model.max_position_embeddings:
        raise ConfigurationError("decode exceeds the model context limit")
    owner = next(o for o in runtime.plan.request_owners if o.phase == "decode")
    replica = runtime.plan.replica_for_instance(owner.domain, owner.instance)
    blocks = -(-peak_context // runtime.plan.block_size)
    if min(args.users, args.requests) * blocks > replica.logical_gpu_blocks:
        raise ConfigurationError("the active-user KV reservation does not fit")
    expected = list(range(4)) + list(range(8, 40))
    if sorted(runtime.placement.device_index_by_worker.values()) != expected:
        raise ConfigurationError("expected A slots 0-3 and F slots 8-39")
    return runtime
