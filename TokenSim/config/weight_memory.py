"""Stored compressed transformer tensors and their actual TP/EP sharding.

Shapes follow vLLM 30f5c01d4ebb: attention.py, compressor.py, nvidia/model.py
and nvidia/mtp.py. This accounts for weights and quantization metadata, not KV,
activations, graph pools, allocator fragmentation or communication workspaces.
"""

from __future__ import annotations

from collections import defaultdict
from math import ceil, prod
from typing import TYPE_CHECKING, Any

from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import dtype_bytes
from TokenSim.moe.placement import ExpertPlacement, build_expert_placement

if TYPE_CHECKING:
    from TokenSim.config.model_config import ModelSpec


def packed_weight_memory(
    model: ModelSpec,
    parallel: ParallelConfig | None = None,
    rank: ParallelRankInfo | None = None,
    placement: ExpertPlacement | None = None,
    *,
    include_mtp: bool = True,
    layout_backend: str | None = None,
) -> dict[str, Any]:
    spec = model.components
    if spec is None:
        raise ConfigurationError(
            "compressed transformer tensor accounting requires its structural metadata"
        )
    parallel = parallel or ParallelConfig.default()
    rank = rank or ParallelRankInfo()
    tp, pp = parallel.tensor_parallel_size, parallel.pipeline_parallel_size
    layout_backend = layout_backend or (
        "mega_moe" if parallel.enable_expert_parallel else "fused"
    )
    if layout_backend not in ("mega_moe", "fused"):
        raise ConfigurationError(
            "compressed transformer weight layout must be mega_moe or fused"
        )
    if layout_backend == "mega_moe" and not parallel.enable_expert_parallel:
        raise ConfigurationError(
            "compressed transformer MegaMoE layout requires expert parallelism"
        )
    if pp != 1:
        # All confirmed AgentX points use PP1. Draft embedding ownership and
        # layer placement for PP>1 must be implemented before accepting it.
        raise ConfigurationError(
            "compressed transformer weight accounting currently supports the confirmed PP1 configurations"
        )
    if (
        min(rank.tp_rank, rank.dp_rank, rank.pp_rank) < 0
        or rank.tp_rank >= tp
        or rank.dp_rank >= parallel.data_parallel_size
        or rank.pp_rank != 0
    ):
        raise ConfigurationError(
            "compressed transformer weight rank is outside its parallel configuration"
        )
    if model.num_attention_heads % tp or spec.o_groups % tp:
        raise ConfigurationError(
            "compressed transformer TP must divide attention heads and output groups"
        )
    if model.moe_intermediate_size % tp:
        raise ConfigurationError(
            "compressed transformer TP must divide expert intermediate width"
        )
    placement = placement or build_expert_placement(
        model.moe, parallel, model.num_layers
    )
    if parallel.enable_expert_parallel:
        local_experts = len(placement.experts_for_rank(rank))
        expert_intermediate = model.moe_intermediate_size
    else:
        local_experts = model.moe.num_experts
        expert_intermediate = model.moe_intermediate_size // tp
    # _use_sequence_parallel: shared experts replicate when attention tokens
    # are sequence-sharded under wide EP. Main checkpoints have TP1.
    sequence_parallel = parallel.sequence_parallel_moe(layout_backend == "mega_moe")
    shared_intermediate = model.moe_intermediate_size * model.moe.num_shared_experts
    if not sequence_parallel:
        block_m = spec.weight_block_size[0]
        shared_intermediate = ceil(shared_intermediate / (block_m * tp)) * block_m

    h, d = model.hidden_size, model.head_dim
    tensors = []
    components: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "parameter_elements": 0,
            "weight_bytes": 0,
            "scale_bytes": 0,
            "metadata_bytes": 0,
            "tensor_count": 0,
        }
    )

    def add(
        name: str,
        component: str,
        shape: tuple[int, ...],
        operator: str,
        *,
        storage_dtype: str | None = None,
        metadata: bool = False,
    ) -> None:
        dtype = storage_dtype or model.operator_dtype(operator)
        elements = prod(shape)
        scale_shape = None
        if dtype in ("fp8", "fp8_block") and len(shape) >= 2:
            scale_shape = (
                *shape[:-2],
                ceil(shape[-2] / spec.weight_block_size[0]),
                ceil(shape[-1] / spec.weight_block_size[1]),
            )
        elif dtype in ("mxfp4", "mxfp4_wo"):
            scale_shape = (*shape[:-1], ceil(shape[-1] / spec.expert_scale_group_size))
        element_bytes = {"int64": 8, "int32": 4}.get(dtype)
        if element_bytes is None:
            element_bytes = dtype_bytes(dtype)
        weight_bytes = ceil(elements * element_bytes)
        scale_bytes = prod(scale_shape) if scale_shape is not None else 0
        stats = components[component]
        stats["parameter_elements"] += 0 if metadata else elements
        stats["weight_bytes"] += 0 if metadata else weight_bytes
        stats["metadata_bytes"] += weight_bytes if metadata else 0
        stats["scale_bytes"] += scale_bytes
        stats["tensor_count"] += 1
        tensors.append(
            {
                "name": name,
                "component": component,
                "shape": list(shape),
                "dtype": dtype,
                "parameter_elements": 0 if metadata else elements,
                "weight_bytes": 0 if metadata else weight_bytes,
                "metadata_bytes": weight_bytes if metadata else 0,
                "scale_shape": list(scale_shape) if scale_shape is not None else None,
                "scale_storage_dtype": "ue8m0" if scale_shape else None,
                "scale_bytes": scale_bytes,
            }
        )

    def compressor(prefix: str, ratio: int, width: int, component: str):
        coff = 2 if ratio == spec.indexed_compress_ratio else 1
        add(prefix + ".fused_wkv_wgate", component, (2 * coff * width, h), "compressor")
        add(prefix + ".norm", component, (width,), "norm")
        add(
            prefix + ".ape",
            component,
            (ratio, coff * width),
            "mhc",
            storage_dtype="fp32",
        )

    def hc_head(prefix: str):
        add(prefix + ".fn", "mhc", (spec.hc_mult, spec.hc_mult * h), "mhc")
        add(prefix + ".base", "mhc", (spec.hc_mult,), "mhc")
        add(prefix + ".scale", "mhc", (1,), "mhc")

    # Vocabulary tensor padding is by 64 before the TP partition. MTP aliases
    # the target embeddings and lm_head after load; it adds neither copy.
    vocab_padded = ceil(model.vocab_size / 64) * 64
    if vocab_padded % tp:
        raise ConfigurationError(
            "compressed transformer padded vocabulary must divide TP"
        )
    add("embedding", "embedding", (vocab_padded // tp, h), "embedding")
    if not model.tie_word_embeddings:
        add("lm_head", "lm_head", (vocab_padded // tp, h), "lm_head")
    add("final_norm", "norm", (h,), "norm")
    hc_head("hc_head")

    for layer in model.layer_specs(include_mtp=include_mtp):
        prefix = f"{'mtp' if layer.is_mtp else 'layers'}.{layer.layer_id}"
        add(
            prefix + ".attn.fused_wqa_wkv",
            "attention",
            (spec.q_lora_rank + d, h),
            "attention_gemm",
        )
        add(
            prefix + ".attn.wq_b",
            "attention",
            (model.q_dim // tp, spec.q_lora_rank),
            "attention_gemm",
        )
        # wo_a is a grouped BMM: input width is q_dim / o_groups, not q_dim.
        add(
            prefix + ".attn.wo_a",
            "attention",
            (spec.o_groups * spec.o_lora_rank // tp, model.q_dim // spec.o_groups),
            "attention_gemm",
        )
        add(
            prefix + ".attn.wo_b",
            "attention",
            (h, spec.o_groups * spec.o_lora_rank // tp),
            "attention_gemm",
        )
        add(
            prefix + ".attn.sink",
            "attention",
            (max(8, model.num_attention_heads // tp),),
            "mhc",
            storage_dtype="fp32",
        )
        for name, width in (
            ("attn.q_norm", spec.q_lora_rank),
            ("attn.kv_norm", d),
            ("attn_norm", h),
            ("ffn_norm", h),
        ):
            add(prefix + "." + name, "norm", (width,), "norm")
        if layer.compress_ratio > 1:
            compressor(
                prefix + ".attn.compressor", layer.compress_ratio, d, "compressor"
            )
        if layer.attention_kind == "csa":
            add(
                prefix + ".indexer.wq_b",
                "indexer",
                (spec.index_n_heads * spec.index_head_dim, spec.q_lora_rank),
                "indexer_gemm",
            )
            add(
                prefix + ".indexer.weights_proj",
                "indexer",
                (spec.index_n_heads, h),
                "indexer_weights_proj",
            )
            compressor(
                prefix + ".indexer.compressor", 4, spec.index_head_dim, "indexer"
            )
        add(prefix + ".ffn.gate", "router", (model.moe.num_experts, h), "router")
        if layer.moe_routing == "hash":
            add(
                prefix + ".ffn.tid2eid",
                "hash_routing",
                (model.vocab_size, model.moe.num_experts_per_tok),
                "router",
                storage_dtype="int64" if layout_backend == "mega_moe" else "int32",
                metadata=True,
            )
        else:
            add(
                prefix + ".ffn.bias",
                "router",
                (model.moe.num_experts,),
                "router",
                storage_dtype="fp32",
            )
        if shared_intermediate:
            add(
                prefix + ".shared.gate_up",
                "shared_expert",
                (2 * shared_intermediate, h),
                "shared_expert",
            )
            add(
                prefix + ".shared.down",
                "shared_expert",
                (h, shared_intermediate),
                "shared_expert",
            )
        add(
            prefix + ".experts.gate_up",
            "routed_expert",
            (local_experts, 2 * expert_intermediate, h),
            "routed_expert",
        )
        add(
            prefix + ".experts.down",
            "routed_expert",
            (local_experts, h, expert_intermediate),
            "routed_expert",
        )
        mix_hc = spec.hc_mult * (2 + spec.hc_mult)
        for site in ("attn", "ffn"):
            add(prefix + f".hc_{site}.fn", "mhc", (mix_hc, spec.hc_mult * h), "mhc")
            add(prefix + f".hc_{site}.base", "mhc", (mix_hc,), "mhc")
            add(prefix + f".hc_{site}.scale", "mhc", (3,), "mhc")
        if layer.is_mtp:
            add(prefix + ".e_proj", "mtp_input_projection", (h, h), "attention_gemm")
            add(prefix + ".h_proj", "mtp_input_projection", (h, h), "attention_gemm")
            for name in ("enorm", "hnorm", "shared_head_norm"):
                add(prefix + "." + name, "norm", (h,), "norm")
            hc_head(prefix + ".hc_head")

    totals = {
        key: sum(value[key] for value in components.values())
        for key in (
            "parameter_elements",
            "weight_bytes",
            "scale_bytes",
            "metadata_bytes",
            "tensor_count",
        )
    }
    totals["total_bytes"] = (
        totals["weight_bytes"] + totals["scale_bytes"] + totals["metadata_bytes"]
    )
    return {
        **totals,
        "components": dict(components),
        "tensors": tensors,
        "local_experts": local_experts,
        "moe_tensor_parallel_size": 1 if parallel.enable_expert_parallel else tp,
        "shared_experts_replicated_across_tp": sequence_parallel,
        "shared_experts_replicated_across_dp": True,
        "layout_backend": layout_backend,
        "include_mtp": include_mtp,
        "mtp_embedding_sharing": "target aliases under PP1; no extra storage",
        "mtp_lm_head_sharing": "target aliases; no extra storage",
        "storage_scope": "persistent weight tensors and quantization metadata; runtime buffers accounted separately",
        "packing_assumption": "FP8 UE8M0 block scales and MXFP4 1x32 UE8M0 scales; transient repacking excluded",
    }
