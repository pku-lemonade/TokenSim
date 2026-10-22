"""Expand evaluated component costs without repeating latency queries."""

from dataclasses import dataclass, field
import math

from TokenSim.config.cache_config import stage_layer_range
from TokenSim.errors import SimulationStateError


@dataclass(frozen=True)
class Segment:
    operation: str
    seconds: float
    layer: int
    category: str = "compute"
    source: str = "composed"
    match: str = "mixed"
    details: dict = field(default_factory=dict)


def _primitive_keys(backend, values):
    model = backend.model
    tokens, hidden = values["T"], model.hidden_size

    def gemm(rows, width, inner, role):
        return "gemm", dict(m=rows, n=width, k=inner, dtype=model.operator_dtype(role))

    def elementwise(operation, rows=tokens, width=hidden):
        return "elementwise", dict(
            op_name=operation,
            num_tokens=int(rows),
            hidden_size=width,
            dtype=model.activation_dtype,
        )

    return {
        "attn_norm_us": elementwise("rmsnorm", values["norm_rows"]),
        "attn_residual_us": elementwise("residual_add", values["forward_rows"]),
        "q_norm_us": elementwise(
            "rmsnorm", tokens * backend.q_dim_local // model.head_dim, model.head_dim
        ),
        "k_norm_us": elementwise(
            "rmsnorm", tokens * backend.kv_dim_local // model.head_dim, model.head_dim
        ),
        "value_norm_us": elementwise(
            "rmsnorm", tokens * backend.kv_heads_local, model.head_dim
        ),
        "rope_us": elementwise(
            "rope", tokens, backend.q_dim_local + backend.kv_dim_local
        ),
        "attn_output_norm_us": elementwise("rmsnorm"),
        "ffn_norm_us": elementwise("rmsnorm"),
        "ffn_residual_us": elementwise("residual_add"),
        "ffn_up_us": gemm(
            tokens,
            model.ffn_projection_count * backend.inter_local,
            hidden,
            "dense_ffn",
        ),
        "ffn_down_us": gemm(tokens, hidden, backend.inter_local, "dense_ffn"),
        "ffn_activation_us": elementwise(model.activation, tokens, backend.inter_local),
        "ffn_output_norm_us": elementwise("rmsnorm"),
        "moe_norm_us": elementwise("rmsnorm", values["norm_rows"]),
        "moe_residual_us": elementwise("residual_add", values["forward_rows"]),
        "router_us": gemm(
            values["router_rows"], model.moe.num_experts, hidden, "router"
        ),
        "layer_scale_us": elementwise("scale"),
        "embedding_scale_us": elementwise("scale"),
        "final_norm_us": elementwise("rmsnorm", values["forward_rows"]),
        "logits_gemm_us": gemm(
            max(1, values["requests"]), backend.vocab_local, hidden, "lm_head"
        ),
        "logit_softcap_us": elementwise(
            "logit_softcap", max(1, values["requests"]), backend.vocab_local
        ),
    }


def expand_composition(backend, values, communications):
    """Order authoritative costs by their model positions, preserving fusion."""
    model = backend.model
    segments = []
    layers = getattr(backend, "trace_layer_indices", None)
    if layers is None:
        layers = tuple(
            stage_layer_range(
                model.num_layers,
                backend.parallel_config.pipeline_parallel_size,
                backend.rank_info.pp_rank,
                backend.pipeline_layer_counts,
            )
        )
    moe_layers = set(model.moe_layer_indices()) if model.is_moe else set()
    comm = {row["kind"]: row for row in communications}
    primitive_keys = _primitive_keys(backend, values)
    tp_count = backend.layers_local + backend.dense_layers_local
    if not backend.ep_enabled:
        tp_count += backend.moe_layers_local

    def add(
        operation, us, layer, category="compute", *, table=None, key=None, details=None
    ):
        if not us:
            return
        estimate = (
            backend._cache.get((table, tuple(sorted(key.items()))))
            if table and key
            else None
        )
        segments.append(
            Segment(
                operation,
                us * 1e-6,
                layer,
                category,
                (
                    estimate.source_id
                    if estimate
                    else f"{backend.fallback}:{table or operation}"
                ),
                (
                    estimate.match_type
                    if estimate
                    else (
                        "analytical"
                        if backend.fallback == "analytical_only"
                        else "composed"
                    )
                ),
                dict(
                    details or {},
                    dtype=model.activation_dtype,
                    tokens=values["T"],
                    shape=key or {},
                    granularity="module" if "module" in operation else "operator",
                ),
            )
        )

    def scalar(operation, key, layer):
        table, shape = primitive_keys.get(key, (None, None))
        add(operation, values.get(key, 0), layer, table=table, key=shape)

    def collective(kind, operation, layer, count, phase=None):
        row = comm.get(kind)
        if row is None or not count:
            return
        estimate = row["estimate"]
        us = row["seconds"] * 1e6 / count
        if phase is not None:
            phases = estimate.detail.get("phase_latency_us")
            if phases is None:
                raise SimulationStateError(
                    "EP timeline requires dispatch/combine cost provenance"
                )
            us = phases[phase]
        add(
            operation,
            us,
            layer,
            "collective",
            details={
                "participants": row["group"],
                "payload_bytes": row["bytes"] // count,
                "latency_source": estimate.source_id,
                "match_type": estimate.match_type,
            },
        )

    scalar("step_overhead", "overhead_us", -1)
    scalar("embedding_scale", "embedding_scale_us", -1)
    projections = list(
        zip(backend.native_attention_projections, values.get("projection_us", ()))
    )
    inputs = [
        (p, us) for p, us in projections if p.name not in ("output", "wo_a", "wo_b")
    ]
    outputs = [(p, us) for p, us in projections if p.name in ("output", "wo_a", "wo_b")]
    layer_specs = (
        model.layer_specs(include_mtp=False) if backend.native_attention else None
    )
    for layer in layers:
        scalar("attention_norm", "attn_norm_us", layer)
        scalar("mhc_module", "mhc_us", layer)
        for projection, us in inputs:
            add(
                projection.name,
                us,
                layer,
                table=projection.table,
                key=projection.key(values["forward_rows"]),
            )
        if not projections:
            scalar("qkv", "qkv_us", layer)
        for operation, key in (
            ("q_norm", "q_norm_us"),
            ("k_norm", "k_norm_us"),
            ("value_norm", "value_norm_us"),
            ("rope", "rope_us"),
        ):
            scalar(operation, key, layer)
        if backend.native_attention:
            kind = layer_specs[layer].attention_kind
            cost = backend.native_attention.trace_layer_costs[kind]
            add("attention_module", cost, layer, table=backend.native_attention.table)
        else:
            add(
                "attention",
                values["attention_us"],
                layer,
                details={
                    "query_context_lengths": tuple(
                        (req.query_tokens, req.num_computed_tokens + req.query_tokens)
                        for req in values["requests_list"]
                    ),
                },
            )
        for projection, us in outputs:
            add(
                projection.name,
                us,
                layer,
                table=projection.table,
                key=projection.key(values["forward_rows"]),
            )
        if not projections:
            scalar("attention_output", "attn_output_us", layer)
        scalar("attention_output_norm", "attn_output_norm_us", layer)
        collective("tp_collective", "all_reduce", layer, tp_count)
        scalar("attention_residual", "attn_residual_us", layer)
        if layer in moe_layers:
            scalar("ffn_norm", "moe_norm_us", layer)
            scalar("router", "router_us", layer)
            collective(
                "ep_all2all", "dispatch", layer, backend.moe_layers_local, "dispatch"
            )
            add(
                "moe_module",
                values["moe_us"] / backend.moe_layers_local
                - values["draft"].get("moe", 0) / backend.moe_layers_local,
                layer,
                table=backend.native_moe.table if backend.native_moe else "moe",
                details={"includes_dispatch_combine": bool(backend.native_moe)},
            )
            collective(
                "ep_all2all", "combine", layer, backend.moe_layers_local, "combine"
            )
            if not backend.ep_enabled:
                collective("tp_collective", "all_reduce", layer, tp_count)
            scalar("ffn_residual", "moe_residual_us", layer)
        else:
            scalar("ffn_norm", "ffn_norm_us", layer)
            scalar("ffn_up", "ffn_up_us", layer)
            scalar(model.activation, "ffn_activation_us", layer)
            scalar("ffn_down", "ffn_down_us", layer)
            scalar("ffn_output_norm", "ffn_output_norm_us", layer)
            collective("tp_collective", "all_reduce", layer, tp_count)
            scalar("ffn_residual", "ffn_residual_us", layer)
        scalar("layer_scale", "layer_scale_us", layer)
    scalar("residual_output_module", "residual_output_us", -1)
    scalar("final_norm", "final_norm_us", -1)
    scalar("lm_head", "logits_gemm_us", -1)
    scalar("logit_softcap", "logit_softcap_us", -1)
    add("mtp_module", sum(values["draft"].values()), -1)
    collective("pp_stage_transfer", "send_recv", -1, 1)
    expected = values["compute_s"] + values["comm_s"]
    if not math.isclose(
        sum(segment.seconds for segment in segments),
        expected,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise SimulationStateError(
            f"operator trace accounting mismatch: expanded={sum(s.seconds for s in segments)} expected={expected}"
        )
    return segments
