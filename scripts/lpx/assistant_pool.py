"""Explicit shared-target-KV Gemma assistant work for greedy draft proposals."""

from dataclasses import dataclass

from TokenSim.af.runtime.kv_sequence import visible_tokens
from TokenSim.config.validation import integer
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError
from scripts.lpx.io_pool import column_shards
from scripts.lpx.assistant_workspace import remote_workspace


@dataclass(frozen=True)
class AssistantSpec:
    hidden: int
    intermediate: int
    backbone: int
    vocab: int
    heads: int
    dims: tuple
    windows: tuple
    kv_heads: tuple

    def __post_init__(self):
        for name in ("hidden", "intermediate", "backbone", "vocab", "heads"):
            integer(getattr(self, name), f"assistant {name}", minimum=1)
        if (
            not self.dims
            or len({len(self.dims), len(self.windows), len(self.kv_heads)}) != 1
        ):
            raise ConfigurationError("assistant layer geometry is incomplete")
        for dim, window, kv in zip(self.dims, self.windows, self.kv_heads):
            integer(dim, "assistant head dimension", minimum=1)
            integer(window, "assistant attention window")
            integer(kv, "assistant KV heads", minimum=1)
            if self.heads % kv:
                raise ConfigurationError(
                    "assistant query heads must divide into KV groups"
                )

    @classmethod
    def from_config(cls, config, target):
        try:
            return cls._from_config(config, target)
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(
                "invalid shared-KV assistant configuration"
            ) from exc

    @classmethod
    def _from_config(cls, config, target):
        text = config["text_config"]
        kinds = text["layer_types"]
        if (
            config["use_ordered_embeddings"]
            or not text["tie_word_embeddings"]
            or text["num_kv_shared_layers"] != len(kinds)
            or text["num_hidden_layers"] != len(kinds)
            or text["num_attention_heads"] != target.num_attention_heads
            or config["backbone_hidden_size"] != target.hidden_size
            or text["vocab_size"] != target.vocab_size
            or set(kinds) != {"sliding_attention", "full_attention"}
            or text.get("enable_moe_block")
        ):
            raise ConfigurationError("unsupported shared-KV assistant architecture")
        dims = tuple(
            text["global_head_dim"] if kind == "full_attention" else text["head_dim"]
            for kind in kinds
        )
        windows = tuple(
            0 if kind == "full_attention" else text["sliding_window"] for kind in kinds
        )
        kv_heads = tuple(
            (
                text["num_global_key_value_heads"]
                if kind == "full_attention"
                else text["num_key_value_heads"]
            )
            for kind in kinds
        )
        spec = cls(
            text["hidden_size"],
            text["intermediate_size"],
            config["backbone_hidden_size"],
            text["vocab_size"],
            text["num_attention_heads"],
            dims,
            windows,
            kv_heads,
        )
        for dim, window, kv in zip(spec.dims, spec.windows, spec.kv_heads):
            matches = [
                i
                for i in range(target.num_layers)
                if (target.attention_window(i) or 0) == window
            ]
            if not matches:
                raise ConfigurationError(
                    "assistant attention has no matching target KV"
                )
            geometry = target.attention_geometry(matches[-1])
            if (dim, kv) != (geometry.head_dim, geometry.num_key_value_heads):
                raise ConfigurationError(
                    "assistant and target shared KV geometry differs"
                )
        return spec

    def parameters(self):
        return dict(
            tied_embedding_lm_head=self.hidden * self.vocab,
            attention_q_and_o=sum(2 * self.hidden * self.heads * d for d in self.dims),
            ffn=3 * len(self.dims) * self.hidden * self.intermediate,
            pre_projection=2 * self.backbone * self.hidden,
            post_projection=self.hidden * self.backbone,
            layer_norms=4 * len(self.dims) * self.hidden,
            final_norm=self.hidden,
            query_norms=sum(self.dims),
            layer_scalars=len(self.dims),
        )


def assistant_memory(
    spec, tp, dtype, *, slots, resident, verification_queries=1, reserve=5_000_000
):
    integer(reserve, "assistant reserve bytes")
    integer(verification_queries, "assistant verifier feedback rows", minimum=1)
    for value, name in ((tp, "TP"), (slots, "slots"), (resident, "resident requests")):
        integer(value, f"assistant {name}", minimum=1)
    if dtype not in {"fp8", "bf16"} or any(
        width % tp for width in (spec.hidden, spec.intermediate, spec.heads)
    ):
        raise ConfigurationError("assistant needs divisible TP and FP8/BF16 matrices")
    counts = spec.parameters()
    matrix_names = ("attention_q_and_o", "ffn", "pre_projection", "post_projection")
    matrix = sum(counts[name] for name in matrix_names) // tp
    nonmatrix = sum(
        counts[name]
        for name in ("layer_norms", "final_norm", "query_norms", "layer_scalars")
    )
    rows = []
    for rank, columns in enumerate(column_shards(spec.vocab, tp)):
        weights = (
            int((matrix + columns * spec.hidden) * dtype_size_bytes(dtype))
            + 2 * nonmatrix
        )
        # Independent slot scratch, communication and persistent request feedback.
        live = (
            6 * columns
            + 12 * spec.hidden
            + 12 * spec.intermediate // tp
            + 12 * spec.heads // tp * max(spec.dims)
            + 8 * spec.backbone
        )
        live = max(live, verification_queries * (2 * spec.backbone + 16))
        communication = 4 * spec.vocab
        persistent = resident * 2 * spec.backbone
        total = weights + slots * (live + communication) + persistent + reserve
        rows.append(
            dict(
                rank=rank,
                weights_bytes=weights,
                live_per_slot_bytes=live,
                communication_per_slot_bytes=communication,
                slots=slots,
                persistent_feedback_bytes=persistent,
                reserve_bytes=reserve,
                total_bytes=total,
                headroom_bytes=500_000_000 - total,
            )
        )
    if any(row["headroom_bytes"] < 0 for row in rows):
        raise ConfigurationError("shared-KV assistant exceeds SRAM")
    return rows


def intersections(source_widths, destination_widths):
    """Nonempty intervals (source rank, destination rank, overlapping heads)."""
    result = []
    source_start = 0
    for source, width in enumerate(source_widths):
        destination_start = 0
        for destination, other in enumerate(destination_widths):
            overlap = min(source_start + width, destination_start + other) - max(
                source_start, destination_start
            )
            if overlap > 0:
                result.append((source, destination, overlap))
            destination_start += other
        source_start += width
    if sum(source_widths) != sum(destination_widths):
        raise ConfigurationError("assistant query/output head intervals differ")
    return result


def element(program, parents, chip, operation, tokens, width, **extra):
    return program.compute(
        parents,
        chip,
        "elementwise",
        dict(
            op_name=operation,
            dtype="bf16",
            num_tokens=tokens,
            hidden_size=width,
            **extra,
        ),
    )


def gemm(program, parents, chip, dtype, n, k):
    return program.compute(parents, chip, "gemm", dict(dtype=dtype, m=1, n=n, k=k))


def remote_attention(
    program,
    parents,
    *,
    spec,
    layer,
    draft_chips,
    target_chips,
    kvp,
    global_kv_layout="separate",
):
    """Run QK/softmax/AV at the existing target cache owners, then merge at draft TP."""
    dim, window = spec.dims[layer], spec.windows[layer]
    target_tp, draft_tp = len(target_chips), len(draft_chips)
    context_width = 1 if window else kvp
    attention_tp = target_tp // context_width
    if spec.heads % attention_tp or spec.kv_heads[layer] % attention_tp:
        raise ConfigurationError("assistant cache-owner head geometry differs")
    target_heads = spec.heads // attention_tp
    local_heads = spec.heads // draft_tp
    cuts = intersections([local_heads] * draft_tp, [target_heads] * attention_tp)
    outbound = [
        (
            draft_chips[source],
            target_chips[destination * context_width + cp],
            heads * dim * 2,
        )
        for source, destination, heads in cuts
        for cp in range(context_width)
    ]
    arrived = program.transfer(parents, outbound)
    attended = []
    for rank, chip in enumerate(target_chips):
        keys = visible_tokens(
            101000, window or None, context_width, rank % context_width, 16
        )
        attended.append(
            program.compute(
                (arrived,),
                chip,
                "generation_attention",
                dict(
                    attn_dtype="bf16",
                    kv_cache_dtype="fp8",
                    num_heads=target_heads,
                    num_kv_heads=spec.kv_heads[layer] // attention_tp,
                    head_dim=dim,
                    window_size=0,
                    batch_size=1,
                    context_len=keys,
                    **(
                        {"kv_cache_layout": global_kv_layout}
                        if not window and global_kv_layout != "separate"
                        else {}
                    ),
                ),
            )
        )
    returned = program.transfer(
        tuple(attended),
        [
            (
                target_chips[destination * context_width + cp],
                draft_chips[source],
                heads * (dim + 1) * 4 if context_width > 1 else heads * dim * 2,
            )
            for source, destination, heads in cuts
            for cp in range(context_width)
        ],
    )
    if context_width == 1:
        return returned
    merged = [
        element(
            program,
            (returned,),
            chip,
            "other_residual",
            context_width * local_heads,
            dim + 1,
            lpx_vector_operation="lse_merge",
            kv_parallel_size=context_width,
        )
        for chip in draft_chips
    ]
    return program.append(tuple(merged))


def draft_step(
    program,
    parents,
    *,
    spec,
    dtype,
    draft_chips,
    head_chips,
    target_chips,
    kvp,
    launch_resource,
    launch_seconds,
    global_kv_layout="separate",
):
    """One exact full-vocabulary greedy proposal, excluding acceptance assumptions."""
    h, backbone, width = spec.hidden, spec.backbone, len(draft_chips)
    launched = program.append(parents, (launch_resource,), duration=launch_seconds)
    acquired = program.acquire((launched,), "embedding-logits-pool/arena")
    lookups = [
        element(program, (acquired,), chip, "cast_fp8_to_bf16", 1, backbone)
        for chip in head_chips
    ]
    lookups = [
        element(program, (node,), chip, "other_residual", 1, backbone)
        for chip, node in zip(head_chips, lookups)
    ]
    embedded = program.reduce(lookups, head_chips, backbone * 2)
    arrived = program.transfer(
        (embedded,),
        [
            (head_chips[i % len(head_chips)], chip, backbone * 2)
            for i, chip in enumerate(draft_chips)
        ],
    )
    program.release_at(acquired, arrived)
    projected = [
        gemm(program, (arrived,), chip, dtype, h // width, 2 * backbone)
        for chip in draft_chips
    ]
    ready = program.reduce(projected, draft_chips, h * 2)
    local_heads = spec.heads // width
    for layer, dim in enumerate(spec.dims):
        normalized = [
            element(program, (ready,), chip, "rmsnorm", 1, h) for chip in draft_chips
        ]
        queries = [
            gemm(program, (node,), chip, dtype, local_heads * dim, h)
            for chip, node in zip(draft_chips, normalized)
        ]
        queries = [
            element(program, (node,), chip, "rmsnorm", local_heads, dim)
            for chip, node in zip(draft_chips, queries)
        ]
        queries = [
            element(program, (node,), chip, "rope", 1, local_heads * dim)
            for chip, node in zip(draft_chips, queries)
        ]
        attended = remote_attention(
            program,
            queries,
            spec=spec,
            layer=layer,
            draft_chips=draft_chips,
            target_chips=target_chips,
            kvp=kvp,
            global_kv_layout=global_kv_layout,
        )
        projected = [
            gemm(program, (attended,), chip, dtype, h, local_heads * dim)
            for chip in draft_chips
        ]
        ready = program.reduce(projected, draft_chips, h * 2)
        intermediate = []
        for chip in draft_chips:
            node = element(program, (ready,), chip, "rmsnorm", 1, h)
            node = element(program, (node,), chip, "residual_add", 1, h)
            node = element(program, (node,), chip, "rmsnorm", 1, h)
            node = gemm(
                program, (node,), chip, dtype, 2 * spec.intermediate // width, h
            )
            node = element(
                program, (node,), chip, "gelu", 1, spec.intermediate // width
            )
            intermediate.append(
                gemm(program, (node,), chip, dtype, h, spec.intermediate // width)
            )
        ready = program.reduce(intermediate, draft_chips, h * 2)
        done = []
        for chip in draft_chips:
            node = element(program, (ready,), chip, "rmsnorm", 1, h)
            node = element(program, (node,), chip, "residual_add", 1, h)
            done.append(element(program, (node,), chip, "other_residual", 1, h))
        ready = program.append(tuple(done))
    norms = [element(program, (ready,), chip, "rmsnorm", 1, h) for chip in draft_chips]
    projected = [
        gemm(program, (node,), chip, dtype, backbone, h // width)
        for chip, node in zip(draft_chips, norms)
    ]
    recurrent = program.reduce(projected, draft_chips, backbone * 2)
    selected = []
    for chip, node, columns in zip(
        draft_chips, norms, column_shards(spec.vocab, width)
    ):
        logits = gemm(program, (node,), chip, dtype, columns, h)
        selected.append(element(program, (logits,), chip, "argmax", 1, columns))
    selected = program.reduce(selected, draft_chips, 16, element_bytes=4)
    # Every boundary head shard receives the token ID before its next lookup.
    selected = program.transfer(
        (selected,),
        [(draft_chips[i % width], chip, 8) for i, chip in enumerate(head_chips)],
    )
    return program.append((selected, recurrent))
