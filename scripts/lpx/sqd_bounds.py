"""Conditional decode bounds coupling GPU latency to resident sliding KV.

Every active request keeps all sliding-window KV on LP30 SRAM while its
global attention visits execute on GPUs. Little's law therefore couples
GPU latency to LP30 inventory, even with free communication and scheduling.
Native whole-operator GPU costs are a protocol assumption, not calibration.
"""

from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.af.runtime.topology import kv_head_partitions
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.workload import work_for_table

ROLES = ("global_attention", "global_core")


def sliding_kv_bytes(model, context):
    """Minimum unreplicated payload using the native separate-cache ledger."""
    integer(context, "SQD context", minimum=1)
    if model.kv_cache_dtype != "fp8":
        raise ConfigurationError("SQD sliding payload requires FP8 cache")
    total = 0
    for layer in range(model.num_layers):
        window = model.attention_window(layer)
        if not window:
            continue
        geometry = model.attention_geometry(layer)
        partition = kv_head_partitions(
            num_kv_heads=geometry.num_key_value_heads,
            tensor_parallel_size=1,
            head_dim=geometry.head_dim,
            num_layers=1,
            block_size=16,
            dtype_size_bytes=1,
        )[0]
        total += min(context, window) * partition.bytes_per_token
    return total


def gpu_global_cost(provider, device, tp, batch, context, role):
    """Reuse native components; core-only ownership excludes GPU projections."""
    model = provider.model
    for name, value in (("TP", tp), ("batch", batch), ("context", context)):
        integer(value, f"SQD {name}", minimum=1)
    if (
        role not in ROLES
        or model.architecture != "gemma4_text"
        or model.is_moe
        or model.components
        or model.dtype != "fp8"
        or model.kv_cache_dtype != "fp8"
        or model.activation_dtype != "bf16"
        or model.global_kv_layout != "separate"
        or model.num_attention_heads % tp
        or provider.hardware.device(device).family != "nvidia_gpu"
        or provider.fallback != "analytical_only"
    ):
        raise ConfigurationError(
            "SQD bounds require native separate-FP8 Gemma GPU costs"
        )
    rows, seconds, payload = [], 0.0, 0.0
    gpu = provider.hardware.device(device)
    efficiency = gpu.analytical.get("memory_efficiency", 0.85)
    number(efficiency, "SQD GPU memory efficiency", positive=True, maximum=1)
    for layer in range(model.num_layers):
        if model.attention_window(layer):
            continue
        query = AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=layer,
            phase="decode",
            stage="attention",
            token_count=batch,
            context_lengths=(context,) * batch,
            model_id=model.model_id,
            dtype="bf16",
            device_type=device,
            worker_id=0,
            local_token_count=batch,
            local_shape=dict(
                tokens=batch,
                hidden_size=model.hidden_size,
                global_hidden_size=model.hidden_size,
                tp_size=tp,
            ),
            weight_dtype="fp8",
            kv_cache_dtype="fp8",
        )
        components = provider.estimate(query).details["components"]
        core = [part for part in components if part["name"] == "attention_core"]
        if len(core) != 1 or core[0]["table"] != "generation_attention":
            raise ConfigurationError("SQD tail proof needs one uniform decode core")
        selected = core if role == "global_core" else components
        elapsed = sum(part["latency_us"] for part in selected) * 1e-6
        seconds += elapsed
        # Core traffic is exactly linear in independent decode batch width.
        payload += work_for_table(core[0]["table"], core[0]["key"]).total_bytes / batch
        rows.append(dict(layer_id=layer, seconds=elapsed, components=list(selected)))
    if not rows:
        raise ConfigurationError("SQD requires at least one global attention layer")
    return dict(
        gpu_path_s=seconds,
        core_memory_s_per_request=payload
        / (gpu.memory_bandwidth_bytes_per_s.value * efficiency),
        layers=rows,
    )


def coupled_bound(
    *,
    useful_flops,
    gpu_path_s,
    gpu_tp,
    batch,
    gpu_peak,
    lp_peak,
    sliding_bytes,
    sram_bytes,
    lp_compute_peak_seconds=0.0,
):
    """Relax weights, buffers, links and integer inventories at any card count.

    For steady token rate T and GPU path latency S, resident requests N >= T*S.
    GPU count G >= T*TP*S/batch and LP30 count L >= N*KV/SRAM.
    LP30 arithmetic supplies an independent lower bound on L, combined by max.
    """
    for name, value in (("GPU TP", gpu_tp), ("batch", batch)):
        integer(value, f"SQD {name}", minimum=1)
    for name, value in (
        ("useful FLOPs", useful_flops),
        ("GPU path", gpu_path_s),
        ("GPU peak", gpu_peak),
        ("LP30 peak", lp_peak),
        ("sliding KV", sliding_bytes),
        ("SRAM capacity", sram_bytes),
    ):
        number(value, f"SQD {name}", positive=True)
    number(lp_compute_peak_seconds, "SQD LP30 compute demand")
    gpu_demand = gpu_peak * gpu_tp * gpu_path_s / batch
    memory_demand = lp_peak * sliding_bytes * gpu_path_s / sram_bytes
    lp_demand = max(memory_demand, lp_compute_peak_seconds)
    return dict(
        mfu_bound_percent=100 * useful_flops / (gpu_demand + lp_demand),
        gpu_peak_seconds_per_token=gpu_demand,
        lp_memory_peak_seconds_per_token=memory_demand,
        lp_compute_peak_seconds_per_token=lp_compute_peak_seconds,
        scope="Necessary steady-decode bound under declared ownership and native GPU costs; all inventories and PP relaxed.",
    )


def finite_inventory_bound(*, max_cards, lp_weight_bytes, **settings):
    """Strengthen the relaxation with LP30 matrix weights and integer inventory.

    GPU HBM capacity, integer resident requests, LP30 shard replication and
    all communication remain relaxed. Unused cards need not be allocated.
    """
    integer(max_cards, "SQD maximum cards", minimum=2)
    number(lp_weight_bytes, "SQD LP30 weights", positive=True)
    coupled_bound(**settings)
    tp, batch = settings["gpu_tp"], settings["batch"]
    latency = settings["gpu_path_s"]
    pg, pl = settings["gpu_peak"], settings["lp_peak"]
    capacity, kv = settings["sram_bytes"], settings["sliding_bytes"]
    arithmetic = settings.get("lp_compute_peak_seconds", 0.0)
    best = None
    # At fixed L the ratio increases until GPU service meets the other limits,
    # then decreases. Check both adjacent GPU groups and the inventory endpoint.
    for lp in range(1, max_cards - tp + 1):
        resident = (lp * capacity - lp_weight_bytes) / kv
        if resident <= 0:
            continue
        other = min(
            resident / latency, lp * pl / arithmetic if arithmetic else float("inf")
        )
        crossing = other * latency / batch
        available = (max_cards - lp) // tp
        candidates = {1, available, int(crossing), int(crossing) + 1}
        for groups in sorted(c for c in candidates if 1 <= c <= available):
            gpu = groups * tp
            tps = min(groups * batch / latency, other)
            mfu = 100 * settings["useful_flops"] * tps / (gpu * pg + lp * pl)
            row = dict(
                mfu_bound_percent=mfu,
                aggregate_tps_bound=tps,
                gpu_cards=gpu,
                lp_cards=lp,
                resident_request_relaxation=resident,
            )
            if best is None or mfu > best["mfu_bound_percent"]:
                best = row
    return best


def batch_tail_bound(*, minimum_batch, core_memory_s_per_request, **settings):
    """Cover every larger integer batch using mandatory linear core traffic."""
    integer(minimum_batch, "SQD tail minimum batch", minimum=1)
    number(core_memory_s_per_request, "SQD tail memory time", positive=True)
    if "batch" in settings or "gpu_path_s" in settings:
        raise ConfigurationError("SQD tail derives batch and latency from memory")
    return coupled_bound(
        **settings,
        batch=minimum_batch,
        gpu_path_s=minimum_batch * core_memory_s_per_request,
    ) | dict(
        minimum_batch=minimum_batch,
        scope="All batches at or above minimum_batch; native linear HBM traffic retained, other GPU work relaxed.",
    )
