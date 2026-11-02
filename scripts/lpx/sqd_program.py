"""Whole-buffer mixed-device decode with bounded LP30 operator scratch."""

from collections import Counter
from functools import lru_cache

from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.config.cache_config import local_kv_heads
from TokenSim.config.validation import integer, number
from TokenSim.errors import ConfigurationError
from scripts.lpx.pool_ops import embedding, logits
from scripts.lpx.program import LP30Program
from scripts.lpx.sqd_bounds import gpu_global_cost


def core_messages(lp, gpu, *, heads, kv_heads, dim, batch=1):
    """Redistribute query heads, one canonical KV delta, and output heads."""
    for name, value in (
        ("heads", heads),
        ("KV heads", kv_heads),
        ("dimension", dim),
        ("batch", batch),
    ):
        integer(value, f"SQD cross {name}", minimum=1)
    if not lp or not gpu or heads % len(lp) or heads % len(gpu) or kv_heads > heads:
        raise ConfigurationError("SQD cross messages require evenly sharded heads")
    for endpoints in (lp, gpu):
        if len(set(endpoints)) != len(endpoints):
            raise ConfigurationError("SQD cross endpoints must be unique")
        for endpoint in endpoints:
            integer(endpoint, "SQD cross endpoint")
    local_kv_heads(kv_heads, len(lp))
    local_kv_heads(kv_heads, len(gpu))
    incoming, outgoing = Counter(), Counter()
    for head in range(heads):
        source = lp[head // (heads // len(lp))]
        target = gpu[head // (heads // len(gpu))]
        incoming[source, target] += batch * dim * 2
        outgoing[target, source] += batch * dim * 2
    for head in range(kv_heads):
        source_rank = (
            head * (len(lp) // kv_heads)
            if len(lp) >= kv_heads
            else head // (kv_heads // len(lp))
        )
        # Head TP beyond the KV-head count requires an append at every replica.
        targets = (
            range(head * (len(gpu) // kv_heads), (head + 1) * (len(gpu) // kv_heads))
            if len(gpu) >= kv_heads
            else (head // (kv_heads // len(gpu)),)
        )
        for rank in targets:
            incoming[lp[source_rank], gpu[rank]] += batch * 2 * dim * 2
    return tuple((a, b, n) for (a, b), n in sorted(incoming.items())), tuple(
        (a, b, n) for (a, b), n in sorted(outgoing.items())
    )


def cross_transfer(program, parents, messages, *, direction, bandwidth, latency_s):
    """One synthetic uplink per LP30 tray and per eight-GPU node, per direction."""
    number(bandwidth, "SQD cross bandwidth", positive=True)
    number(latency_s, "SQD cross latency")
    if direction not in {"to_gpu", "to_lp"} or not messages:
        raise ConfigurationError("SQD cross transfer requires messages and direction")
    loads = Counter()
    for source, target, size in messages:
        integer(source, "SQD source endpoint")
        integer(target, "SQD target endpoint")
        integer(size, "SQD cross payload", minimum=1)
        lp, gpu = (source, target) if direction == "to_gpu" else (target, source)
        ldir, gdir = ("tx", "rx") if direction == "to_gpu" else ("rx", "tx")
        for key in (
            f"sqd/lp:{lp}:{ldir}",
            f"sqd/gpu:{gpu}:{gdir}",
            f"sqd/lp-tray:{lp // 16}:{ldir}",
            f"sqd/gpu-node:{gpu // 8}:{gdir}",
        ):
            loads[key] += size
    sent = program.append(
        parents, tuple(sorted(loads)), duration=max(loads.values()) / bandwidth
    )
    arrived = program.append((sent,), delay=latency_s)
    return sent, arrived


class GlobalCoreProgram:
    """One profile per permanent request/GPU-replica affinity.

    Request-specific hidden and QKV slots are charged by CoreMemory. Scratch
    is acquired only while an LP30 operator and its outgoing data remain live.
    GPU waits retain request slots and release the LP30 scratch arena.
    """

    def __init__(
        self,
        hardware,
        model,
        *,
        lp_tp,
        counts,
        gpu_device,
        gpu_tp,
        replicas,
        batch=1,
        cross_latency_us=2.0,
        cross_bandwidth=200e9,
        collectives=True,
        host_submit_us=4.0,
    ):
        self.hardware, self.model = hardware, model
        try:
            self.counts = tuple(counts)
        except TypeError as exc:
            raise ConfigurationError("SQD stage counts require an iterable") from exc
        self.lp_tp, self.gpu_tp = lp_tp, gpu_tp
        self.replicas, self.batch, self.gpu_device = replicas, batch, gpu_device
        for name, value in (
            ("LP30 TP", lp_tp),
            ("GPU TP", gpu_tp),
            ("replicas", replicas),
            ("batch", batch),
        ):
            integer(value, f"SQD {name}", minimum=1)
        if (
            not self.counts
            or any(type(n) is not int or n < 1 for n in self.counts)
            or sum(self.counts) != model.num_layers
        ):
            raise ConfigurationError("SQD stage counts must cover the model")
        if model.num_attention_heads % lp_tp or model.num_attention_heads % gpu_tp:
            raise ConfigurationError("SQD TP must divide query heads")
        number(cross_latency_us, "SQD cross latency")
        number(cross_bandwidth, "SQD cross bandwidth", positive=True)
        number(host_submit_us, "SQD host submission")
        if type(collectives) is not bool:
            raise ConfigurationError("SQD collective control must be boolean")
        self.collectives = collectives
        self.cross_latency = cross_latency_us * 1e-6
        self.cross_bandwidth = cross_bandwidth
        self.host_submit_s = host_submit_us * 1e-6
        self.provider = OperatorTableAFComputeProvider(
            hardware, model, fallback="analytical_only", graph_device_kernel_us=0.2
        )
        self.parts = lru_cache(maxsize=None)(self.parts)
        self.gpu_cost = gpu_global_cost(
            self.provider, gpu_device, gpu_tp, batch, 101000, "global_core"
        )
        self.lp_cards = lp_tp * len(self.counts) + 3
        self.gpu_cards = len(self.gpu_cost["layers"]) * gpu_tp * replicas

    def parts(self, layer, stage, operation=""):
        model, batch = self.model, self.batch
        query = AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=layer,
            phase="decode",
            stage=stage,
            token_count=batch,
            context_lengths=(101000,) * batch,
            model_id=model.model_id,
            dtype="bf16",
            device_type="groq_lp30",
            worker_id=0,
            operation=operation,
            local_token_count=batch,
            local_shape=dict(
                tokens=batch,
                hidden_size=model.hidden_size,
                global_hidden_size=model.hidden_size,
                tp_size=self.lp_tp,
            ),
            weight_dtype="fp8",
            kv_cache_dtype="fp8",
            source_layout="replicated",
            target_layout="replicated",
        )
        return self.provider.estimate(query).details["components"]

    def compute(self, program, parents, chips, components):
        resources = tuple(f"mfu-search/worker:{chip}" for chip in chips) + tuple(
            f"lp30:{chip}:vxm" for chip in chips
        )
        program.operators.update({part["name"]: len(chips) for part in components})
        return program.append(
            parents,
            resources,
            duration=sum(part["latency_us"] for part in components) * 1e-6,
        )

    def reduce(self, program, parent, chips):
        if not self.collectives:
            return program.append((parent,))
        return program.reduce((parent,), chips, self.batch * self.model.hidden_size * 2)

    def build(self, replica=0):
        integer(replica, "SQD program GPU replica")
        if replica >= self.replicas:
            raise ConfigurationError("SQD GPU replica is unavailable")
        p = LP30Program(self.hardware, self.lp_cards)
        h, batch = self.model.hidden_size, self.batch
        pool = tuple(range(self.lp_cards - 3, self.lp_cards))
        launch = p.append((), ("sqd/host",), duration=self.host_submit_s)
        arena = p.acquire((launch,), "sqd/boundary-scratch")
        ready = embedding(p, (arena,), pool, self.model, batch)
        p.release_at(arena, ready)
        ready = p.transfer(
            (ready,),
            [(pool[rank % 3], rank, batch * h * 2) for rank in range(self.lp_tp)],
        )
        layer, global_index = 0, 0
        for stage, count in enumerate(self.counts):
            chips = tuple(range(stage * self.lp_tp, (stage + 1) * self.lp_tp))
            scratch = f"sqd/lp-scratch:{stage}"
            for _ in range(count):
                parts = self.parts(layer, "attention")
                acquired = p.acquire((ready,), scratch)
                if self.model.attention_window(layer):
                    ready = self.compute(p, (acquired,), chips, parts)
                else:
                    before = [
                        part
                        for part in parts
                        if part["name"] not in {"attention_core", "output_projection"}
                    ]
                    after = [
                        part for part in parts if part["name"] == "output_projection"
                    ]
                    pre = self.compute(p, (acquired,), chips, before)
                    base = (global_index * self.replicas + replica) * self.gpu_tp
                    gpu = tuple(range(base, base + self.gpu_tp))
                    geometry = self.model.attention_geometry(layer)
                    outgoing, incoming = core_messages(
                        chips,
                        gpu,
                        heads=self.model.num_attention_heads,
                        kv_heads=geometry.num_key_value_heads,
                        dim=geometry.head_dim,
                        batch=batch,
                    )
                    sent, arrived = cross_transfer(
                        p,
                        (pre,),
                        outgoing,
                        direction="to_gpu",
                        bandwidth=self.cross_bandwidth,
                        latency_s=self.cross_latency,
                    )
                    p.release_at(acquired, sent)
                    core = p.append(
                        (arrived,),
                        tuple(f"sqd/gpu-worker:{rank}" for rank in gpu),
                        duration=self.gpu_cost["layers"][global_index]["seconds"],
                    )
                    _, returned = cross_transfer(
                        p,
                        (core,),
                        incoming,
                        direction="to_lp",
                        bandwidth=self.cross_bandwidth,
                        latency_s=self.cross_latency,
                    )
                    acquired = p.acquire((returned,), scratch)
                    ready = self.compute(p, (acquired,), chips, after)
                    global_index += 1
                ready = self.reduce(p, ready, chips)
                ready = self.compute(
                    p,
                    (ready,),
                    chips,
                    self.parts(layer, "layout", "post_attention_norm"),
                )
                p.release_at(acquired, ready)
                acquired = p.acquire((ready,), scratch)
                ready = self.compute(
                    p, (acquired,), chips, self.parts(layer, "dense_ffn")
                )
                ready = self.reduce(p, ready, chips)
                ready = self.compute(p, (ready,), chips, self.parts(layer, "residual"))
                p.release_at(acquired, ready)
                layer += 1
            if stage + 1 < len(self.counts):
                ready = p.transfer(
                    (ready,),
                    [(chip, chip + self.lp_tp, batch * h * 2) for chip in chips],
                )
        ready = p.transfer(
            (ready,),
            [
                (chips[rank % len(chips)], target, batch * h * 2)
                for rank, target in enumerate(pool)
            ],
        )
        arena = p.acquire((ready,), "sqd/boundary-scratch")
        ready = logits(p, (arena,), pool, self.model, batch)
        p.release_at(arena, ready)
        profile = p.profile()
        profile.metadata.update(
            lp30_cards=self.lp_cards,
            gpu_cards=self.gpu_cards,
            gpu_device=self.gpu_device,
            replica=replica,
            lp30_stage_counts=self.counts,
            global_core_visits=global_index,
            cross_uplink_bandwidth=self.cross_bandwidth,
            cross_latency_s=self.cross_latency,
            host_submit_s=self.host_submit_s,
            lp30_collectives_priced=self.collectives,
            scope="Native-cost mixed-device whole-buffer prototype with projected KV capacity and per-request activation slots; no production mixed-ownership compiler or physical-placement validation.",
        )
        return profile
