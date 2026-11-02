"""Native-ledger memory projection for GPU global cores and LP30 projections."""

from dataclasses import replace
from functools import lru_cache

from TokenSim.config.cache_config import local_kv_heads
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError
from scripts.lpx.io_pool import pool_memory, transformer_setup
from scripts.lpx.mfu_probe import MFUCase, runtime_for
from scripts.lpx.partition import native_footprint, partition_solutions


class CoreMemory:
    """Retain LP30 weights/scratch and relocate only global KV to GPUs.

    The production compiler does not implement this ownership. The projection
    reuses its layer weights, windows, block rounding and default watermark.
    Per-request activation slots coexist with one reusable compute scratch arena.
    """

    def __init__(self, config, tp, batch=1):
        integer(tp, "SQD LP30 TP", minimum=1)
        integer(batch, "SQD batch", minimum=1)
        self.batch, self.tp = batch, tp
        config, hardware, model, proxy = transformer_setup(
            config, tp=tp, queries=1, batch=batch
        )
        if (
            model.global_kv_layout != "separate"
            or model.num_attention_heads % tp
            or model.kv_cache_dtype != "fp8"
            or model.activation_dtype != "bf16"
        ):
            raise ConfigurationError("SQD projection requires separate KV and head TP")
        seed = runtime_for(
            hardware,
            proxy,
            MFUCase(tp, 1, proxy.num_layers, batch=batch),
            proxy.num_layers,
        )
        native = native_footprint(seed, 102000)
        zero = (0,) * tp
        self.footprint = replace(
            native,
            kv=tuple(
                row if model.attention_window(i) else zero
                for i, row in enumerate(native.kv)
            ),
            blocks=tuple(
                row if model.attention_window(i) else zero
                for i, row in enumerate(native.blocks)
            ),
            embedding=zero,
        )
        maximum_qkv = max(
            (
                model.num_attention_heads // tp
                + 2
                * local_kv_heads(model.attention_geometry(i).num_key_value_heads, tp)
            )
            * model.attention_geometry(i).head_dim
            for i in range(model.num_layers)
        )
        # Four full hidden vectors and two Q/K/V bundles cover request input,
        # residual, partial/reduced output and both directions of staging.
        self.request_slot_bytes = 2 * (4 * model.hidden_size + 2 * maximum_qkv)
        self.hardware, self.model, self.config = hardware, model, config
        self._select = lru_cache(maxsize=None)(self._select)

    def select(self, pp, resident):
        integer(pp, "SQD LP30 PP", minimum=1)
        integer(resident, "SQD resident requests", minimum=1)
        if pp > self.model.num_layers or resident % self.batch:
            raise ConfigurationError(
                "SQD memory requires valid PP and complete cohorts"
            )
        return self._select(pp, resident)

    def _select(self, pp, resident):
        fp = replace(
            self.footprint,
            kv=tuple(tuple(n * resident for n in row) for row in self.footprint.kv),
            buffers=tuple(
                n + resident * self.request_slot_bytes for n in self.footprint.buffers
            ),
        )
        solution = partition_solutions(fp, pp).get(pp)
        if solution is None:
            return None
        boundary = pool_memory(self.model, 3, self.batch)
        for row in boundary:
            extra = resident * 4 * self.model.hidden_size * 2
            row["request_slots_bytes"] = extra
            row["total_bytes"] += extra
            row["headroom_bytes"] -= extra
        if any(row["headroom_bytes"] < 0 for row in boundary):
            return None
        return dict(
            **solution,
            resident_requests=resident,
            activation_slot_bytes_per_request_per_lp30=self.request_slot_bytes,
            reusable_native_buffers_per_lp30=self.footprint.buffers,
            minimum_lp30_headroom_bytes=min(
                500_000_000 - n for row in solution["stage_rank_bytes"] for n in row
            ),
            boundary_pool=boundary,
            scope="Native-ledger projection; mixed global-core ownership and physical memory placement are uncompiled.",
        )

    def maximum(self, pp, limit=256):
        integer(limit, "SQD memory request search limit", minimum=1)
        low, high = 0, limit // self.batch
        while low < high:
            middle = (low + high + 1) // 2
            if self.select(pp, middle * self.batch):
                low = middle
            else:
                high = middle - 1
        return self.select(pp, low * self.batch) if low else None


def gpu_memory(hardware, model, *, device, tp, replicas, resident, batch=1):
    for name, value in (
        ("TP", tp),
        ("replicas", replicas),
        ("resident", resident),
        ("batch", batch),
    ):
        integer(value, f"SQD GPU {name}", minimum=1)
    if (
        model.num_attention_heads % tp
        or hardware.device(device).family != "nvidia_gpu"
        or model.kv_cache_dtype != "fp8"
        or model.activation_dtype != "bf16"
        or resident % batch
    ):
        raise ConfigurationError("SQD GPU memory requires head TP on GPUs")
    gpu = hardware.device(device)
    requests = -(-(resident // batch) // replicas) * batch
    rows = []
    for i in range(model.num_layers):
        if model.attention_window(i):
            continue
        geometry = model.attention_geometry(i)
        kv = local_kv_heads(geometry.num_key_value_heads, tp) * geometry.head_dim
        q = model.num_attention_heads // tp * geometry.head_dim
        cache = requests * 102000 * 2 * kv
        staged = requests * 2 * (2 * q + 4 * kv)
        reserve = int(gpu.memory_reserved_bytes.value)
        total = cache + staged + reserve
        rows.append(
            dict(
                layer_id=i,
                replicas=replicas,
                ranks_per_replica=tp,
                requests_per_busiest_replica=requests,
                kv_bytes_per_rank=cache,
                staged_bytes_per_rank=staged,
                reserve_bytes_per_rank=reserve,
                total_bytes_per_rank=total,
                headroom_bytes_per_rank=int(gpu.memory_capacity_bytes.value) - total,
            )
        )
    if any(row["headroom_bytes_per_rank"] < 0 for row in rows):
        raise ConfigurationError("SQD global KV exceeds GPU HBM capacity")
    return rows
