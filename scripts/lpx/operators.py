"""Compose TokenSim operator services for a layer-partitioned projection.

PP partitions resident layers; CP partitions the attention context, with
replicated projections and an explicit attention-result AllReduce.
The serving experiment reserves a whole A/F pool per operation.
"""

from collections import Counter
from dataclasses import dataclass
import json
import math

import numpy as np

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.af.runtime.memory import _experts_by_ep
from TokenSim.comm.collectives import CollectiveModel
from TokenSim.comm.collective_types import CollectiveQuery
from TokenSim.config.cache_config import local_kv_heads
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import dtype_bytes
from TokenSim.latency.operator_table import OperatorResolver


@dataclass(frozen=True)
class Task:
    name: str
    pool: str
    seconds: float


class Operators:
    def __init__(self, hardware, model):
        self.hardware, self.model = hardware, model
        self.provider = OperatorTableAFComputeProvider(
            hardware, model, fallback="analytical_only"
        )
        self.resolvers = {}
        self.collectives = {}
        self.expert_owners = {}

    def resolver(self, device):
        if device not in self.resolvers:
            self.resolvers[device] = OperatorResolver(
                device=self.hardware.device(device),
                fallback="analytical_only",
                gated=self.model.gated,
            )
        return self.resolvers[device]

    def collective(self, device, ranks, size, operation="all_reduce"):
        if ranks == 1:
            return 0.0
        if device not in self.collectives:
            spec = self.hardware.device(device)
            topo = self.hardware.topology(
                "groq_lpx256" if spec.family == "groq_tsp" else "rubin_nvl72"
            )
            self.collectives[device] = CollectiveModel(
                topo,
                self.hardware.links,
                family=spec.family,
                launch_us=spec.analytical.get("collective_launch_us", 0.0),
                fallback="analytical_only",
            )
        collective = self.collectives[device]
        layout = collective.topology.group_layout(tuple(range(ranks)))
        return collective.estimate(
            CollectiveQuery(operation, size, layout, "bf16")
        ).latency_s

    def query(
        self, stage, device, tp, tokens, batch, context, query_len, layer, operation=""
    ):
        h = self.model.hidden_size
        return AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=layer,
            phase="decode" if query_len == 1 else "prefill",
            stage=stage,
            token_count=tokens,
            context_lengths=(context,) * batch,
            query_lengths=(query_len,) * batch,
            model_id=self.model.model_id,
            dtype=self.model.activation_dtype,
            device_type=device,
            worker_id=0,
            operation=operation,
            local_token_count=tokens,
            local_shape={
                "tokens": tokens,
                "hidden_size": h,
                "global_hidden_size": h,
                "tp_size": tp,
            },
            weight_dtype=self.model.dtype,
            activation_dtype=self.model.activation_dtype,
            kv_cache_dtype=self.model.kv_cache_dtype,
            layer_type="moe",
            tp_group=tuple(range(tp)),
        )

    def resident_bytes(self, layout, batch, context, workspace_tokens=1):
        m = self.model
        if (
            m.num_attention_heads % layout.a_tp
            or m.hidden_size % layout.a_tp
            or m.moe_intermediate_size % layout.f_tp
        ):
            raise ConfigurationError("layout cannot evenly partition model dimensions")
        if max(layout.a_pp, layout.f_pp) > m.num_layers:
            raise ConfigurationError("pipeline stages cannot exceed layer count")
        b_w, b_a, b_k = (
            dtype_bytes(m.dtype),
            dtype_bytes(m.activation_dtype),
            dtype_bytes(m.kv_cache_dtype),
        )
        kv_heads = local_kv_heads(m.num_key_value_heads, layout.a_tp)
        a_layers, f_layers = math.ceil(m.num_layers / layout.a_pp), math.ceil(
            m.num_layers / layout.f_pp
        )
        q_width = m.num_attention_heads // layout.a_tp * m.head_dim
        kv_width = kv_heads * m.head_dim
        # The largest A stage holds embedding/head storage and router weights.
        a_weights = (
            a_layers
            * (
                2 * m.hidden_size * (q_width + kv_width)
                + m.hidden_size * m.moe.num_experts
            )
            + m.embedding_params() / layout.a_tp
        ) * b_w
        kv = batch * math.ceil(context / layout.a_cp) * 2 * kv_width * a_layers * b_k
        f_weights = (
            math.ceil(m.moe.num_experts / layout.f_ep)
            * m.expert_params()
            / layout.f_tp
            * f_layers
            * b_w
        )
        a_workspace = (
            2
            * batch
            * workspace_tokens
            * max(m.hidden_size, m.vocab_size // layout.a_tp)
            * b_a
        )
        f_workspace = (
            2
            * batch
            * workspace_tokens
            * m.moe.num_experts_per_tok
            * max(m.hidden_size, m.moe_intermediate_size // layout.f_tp)
            * b_a
        )
        a_bytes, f_bytes = a_weights + kv + a_workspace, f_weights + f_workspace
        if layout.colocated:
            a_bytes = a_weights + f_weights + kv + max(a_workspace, f_workspace)
            f_bytes = a_bytes
        report = {
            "a_weights": a_weights,
            "f_weights": f_weights,
            "kv": kv,
            "a_workspace": a_workspace,
            "f_workspace": f_workspace,
            "a_required": a_bytes,
            "f_required": f_bytes,
        }
        for role, device, required in (
            ("A", layout.a_device, a_bytes),
            ("F", layout.f_device, f_bytes),
        ):
            capacity = self.hardware.device(device).usable_memory_bytes
            if required > capacity:
                raise ConfigurationError(
                    f"{role} rank memory does not fit on {device}: required={required:.0f}, available={capacity:.0f}"
                )
        return report

    def histogram(self, batch, query_len, seed, layer):
        m = self.model
        if query_len == 1:
            rng = np.random.default_rng(np.random.SeedSequence([seed, layer, batch]))
            counts = Counter(
                int(e)
                for _ in range(batch)
                for e in rng.choice(
                    m.moe.num_experts, m.moe.num_experts_per_tok, replace=False
                )
            )
        else:
            # Deterministic balanced prefill approximation, with exact assignment conservation.
            total = batch * query_len * m.moe.num_experts_per_tok
            base, extra = divmod(total, m.moe.num_experts)
            counts = Counter(
                {
                    e: base + (e < extra)
                    for e in range(m.moe.num_experts)
                    if base or e < extra
                }
            )
        return counts

    def forward(
        self,
        layout,
        pool,
        batch,
        query_len,
        context,
        seed=42,
        logits=True,
        logits_rows=None,
    ):
        m = self.model
        tokens = batch * query_len
        a_pool, f_pool = f"{pool}:a", f"{pool}:a" if layout.colocated else f"{pool}:f"
        tasks = []

        def compute(name, query, target=a_pool):
            seconds = self.provider.estimate(query).service_duration
            tasks.append(Task(name, target, seconds))

        def elem(name, operation, device, width, count=tokens, target=a_pool):
            key = dict(
                op_name=operation,
                dtype=m.activation_dtype,
                num_tokens=count,
                hidden_size=width,
            )
            seconds = self.resolver(device).query("elementwise", key).latency_us / 1e6
            tasks.append(Task(name, target, seconds))

        elem("embedding", "embedding_lookup", layout.a_device, m.hidden_size)
        for layer in range(m.num_layers):
            start = len(tasks)
            a_query = self.query(
                "attention",
                layout.a_device,
                layout.a_tp,
                tokens,
                batch,
                context,
                query_len,
                layer,
            )
            estimate = self.provider.estimate(a_query)
            components = estimate.details["components"]
            a_time = 0.0
            for component in components:
                if component["name"] == "attention_core" and layout.a_cp > 1:
                    key = dict(component["key"])
                    # CP divides the cached prefix; current queries remain local.
                    if query_len == 1:
                        key["context_len"] = math.ceil(context / layout.a_cp)
                    else:
                        key["context_len"] = query_len + math.ceil(
                            (context - query_len) / layout.a_cp
                        )
                    a_time += (
                        self.resolver(layout.a_device)
                        .query(component["table"], key)
                        .latency_us
                        / 1e6
                    )
                else:
                    a_time += component["latency_us"] / 1e6
            a_time += self.collective(
                layout.a_device, layout.a_tp, tokens * m.hidden_size * 2
            )
            a_time += self.collective(
                layout.a_device, layout.a_cp, tokens * m.q_dim / layout.a_tp * 2
            )
            tasks.append(Task(f"layer{layer}:attention", a_pool, a_time))
            elem(f"layer{layer}:post_norm", "rmsnorm", layout.a_device, m.hidden_size)
            compute(
                f"layer{layer}:router",
                self.query(
                    "router",
                    layout.a_device,
                    1,
                    tokens,
                    batch,
                    context,
                    query_len,
                    layer,
                ),
            )
            histogram = self.histogram(batch, query_len, seed, layer)
            if layout.f_ep not in self.expert_owners:
                placement = _experts_by_ep(m.moe.num_experts, layout.f_ep, "linear")
                self.expert_owners[layout.f_ep] = {
                    expert: rank
                    for rank, experts in placement.items()
                    for expert in experts
                }
            groups = {}
            for expert, count in histogram.items():
                group = self.expert_owners[layout.f_ep][expert]
                groups.setdefault(group, []).append(count)
            f_time = 0.0
            for counts in groups.values():
                key = dict(
                    dtype=m.dtype,
                    hidden_size=m.hidden_size,
                    inter_size=m.moe_intermediate_size // layout.f_tp,
                    expert_tokens=json.dumps(sorted(counts), separators=(",", ":")),
                )
                f_time = max(
                    f_time,
                    self.resolver(layout.f_device)
                    .query("routed_experts", key)
                    .latency_us
                    / 1e6,
                )
            f_time += self.collective(
                layout.f_device,
                layout.f_tp,
                max(map(sum, groups.values())) * m.hidden_size * 2,
            )
            # Dispatch and partial-return transport each routed assignment.
            payload = sum(histogram.values()) * m.hidden_size * 2
            net = (
                "ethernet1600Gb"
                if layout.a_device != layout.f_device or not layout.colocated
                else self.hardware.device(layout.a_device).scale_up_link
            )
            link = self.hardware.link(net)
            if layout.f_ep > 1 or not layout.colocated:
                tasks.append(
                    Task(
                        f"layer{layer}:dispatch",
                        f"{pool}:tx",
                        link.analytical_transfer_us(payload) / 1e6,
                    )
                )
            tasks.append(Task(f"layer{layer}:experts", f_pool, f_time))
            if layout.f_ep > 1 or not layout.colocated:
                tasks.append(
                    Task(
                        f"layer{layer}:return",
                        f"{pool}:rx",
                        link.analytical_transfer_us(payload) / 1e6,
                    )
                )
            elem(
                f"layer{layer}:combine",
                "weighted_combine",
                layout.a_device,
                m.hidden_size,
                tokens * m.moe.num_experts_per_tok,
            )
            elem(
                f"layer{layer}:residual", "residual_add", layout.a_device, m.hidden_size
            )
            if layer and (
                layer * layout.a_pp // m.num_layers
                != (layer - 1) * layout.a_pp // m.num_layers
            ):
                # Explicit layer-partition handoff; no assumed compiler overlap.
                tasks.insert(
                    start,
                    Task(
                        f"layer{layer}:pipeline_handoff",
                        f"{pool}:tx",
                        self.hardware.link("ethernet1600Gb").analytical_transfer_us(
                            tokens * m.hidden_size * 2
                        )
                        / 1e6,
                    ),
                )
        if logits:
            rows = batch if logits_rows is None else logits_rows
            elem("final_norm", "rmsnorm", layout.a_device, m.hidden_size, rows)
            compute(
                "lm_head",
                self.query(
                    "layout",
                    layout.a_device,
                    layout.a_tp,
                    rows,
                    rows,
                    context,
                    1,
                    0,
                    "lm_head",
                ),
            )
            tasks.append(
                Task(
                    "logits_gather",
                    f"{pool}:tx",
                    self.collective(
                        layout.a_device,
                        layout.a_tp,
                        rows * m.vocab_size / layout.a_tp * 2,
                        "all_gather",
                    ),
                )
            )
            elem("sample", "argmax", layout.a_device, m.vocab_size, rows)
        return tasks
