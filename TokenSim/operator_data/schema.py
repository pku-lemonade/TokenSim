from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import json
import math
from functools import cached_property

from TokenSim.config.validation import number
from TokenSim.errors import ConfigurationError

SCHEMA_VERSION = 1
LATENCY_UNIT = "us"
VALUE_FIELDS = ("latency_us", "source_id")

COLLECTIVE_OPERATIONS = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "send_recv",
)
EP_ALL2ALL_PHASES = ("dispatch", "combine")
# Names of the expert-parallel all-to-all implementations. The first two map to
# ParallelConfig.all2all_backend; the ``deepep_v2_*`` names are vLLM's split
# context/generation DeepEP variants and are only reachable by explicit query.
EP_ALL2ALL_MODES = (
    "deepep_high_throughput",
    "deepep_low_latency",
    "deepep_v2_context",
    "deepep_v2_generation",
)
# How far past the measured sequence range attention queries may extrapolate
# (LookupPolicy.max_extrapolation_ratio, 16x, applies to every other axis).
ATTENTION_SEQ_EXTRAPOLATION_RATIO = 64.0
ELEMENTWISE_OPS = (
    "rmsnorm",
    "layernorm",
    "rope",
    "residual_add",
    "swiglu",
    "gelu",
    "data_reorder",
    "other_residual",
    "embedding_lookup",
    "argmax",
    "topk",
    "weighted_combine",
    "post_attention_norm",
)


def canonical_histogram(value: Any, *, field: str) -> str:
    """Return a stable JSON representation for length/token histograms.

    Tables are often written by parquet/CSV tooling with ``1`` versus ``1.0``
    or with a different pair order.  Canonicalising at both package ingest and
    query time keeps those representations equivalent and hashable.
    """
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{field} must be a JSON list of [value,count] pairs"
            ) from exc
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{field} must be a non-empty list of [value,count] pairs")
    pairs: list[tuple[int, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{field}[{index}] must contain exactly [value,count]")
        try:
            value_i = int(item[0])
            count_f = float(item[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{index}] values must be numeric") from exc
        if value_i <= 0 or not math.isfinite(count_f) or count_f <= 0:
            raise ValueError(f"{field}[{index}] requires positive value and count")
        pairs.append((value_i, count_f))
    pairs.sort(key=lambda pair: pair[0])
    return json.dumps(pairs, separators=(",", ":"), ensure_ascii=False)


def normalize_table_key_value(table: str, field: str, value: Any) -> Any:
    if table == "mla_attention" and field == "context_lengths":
        return canonical_histogram(value, field=field)
    if table == "routed_experts" and field == "expert_tokens":
        raw = value
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{field} must be a JSON list of positive counts"
                ) from exc
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError(f"{field} must be a non-empty list of positive counts")
        counts: list[int] = []
        for index, item in enumerate(raw):
            try:
                count = int(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field}[{index}] must be an integer") from exc
            if count <= 0:
                raise ValueError(f"{field}[{index}] must be positive")
            counts.append(count)
        return json.dumps(counts, separators=(",", ":"))
    return value


@dataclass(frozen=True)
class AxisSpec:
    """A key field along which controlled interpolation is permitted.

    ``max_extrapolation_ratio`` overrides the lookup policy's bound for this
    axis; ``None`` keeps the policy default. Axes whose value barely changes the
    cost (the expert count of a DeepEP dispatch) can be left effectively
    unbounded because the analytical growth reference is flat along them.
    """

    name: str
    scale: str = "log"  # "log": geometric interpolation (sizes); "linear": arithmetic
    max_extrapolation_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.scale not in {"log", "linear"}:
            raise ConfigurationError(f"axis {self.name!r}: scale must be log or linear")
        if self.max_extrapolation_ratio is not None:
            number(
                self.max_extrapolation_ratio,
                f"axis {self.name!r}: max_extrapolation_ratio",
                minimum=1,
            )


@dataclass(frozen=True)
class TableSpec:
    name: str
    key_fields: tuple[str, ...]
    axes: tuple[AxisSpec, ...]
    dtype_fields: tuple[str, ...]
    description: str
    metadata_fields: tuple[str, ...] = ()

    # Specs are immutable; derived field lists are computed once.

    @cached_property
    def axis_names(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    @cached_property
    def discrete_fields(self) -> tuple[str, ...]:
        return tuple(field for field in self.key_fields if field not in self.axis_names)

    @cached_property
    def required_fields(self) -> tuple[str, ...]:
        return self.key_fields + VALUE_FIELDS

    @property
    def perf_filename(self) -> str:
        return f"{self.name}_perf.parquet"

    @property
    def analysis_filename(self) -> str:
        return f"{self.name}_analysis.parquet"


TABLE_SPECS: Mapping[str, TableSpec] = {
    "batched_gemm": TableSpec(
        name="batched_gemm",
        key_fields=("dtype", "groups", "m", "n", "k"),
        axes=(AxisSpec("m"),),
        dtype_fields=("dtype",),
        description="One grouped launch of equally shaped independent GEMMs.",
    ),
    "mla_attention": TableSpec(
        name="mla_attention",
        key_fields=(
            "attn_dtype",
            "kv_cache_dtype",
            "phase",
            "execution_mode",
            "batch_size",
            "query_len",
            "context_lengths",
            "num_heads",
            "kv_lora_rank",
            "rope_head_dim",
            "nope_head_dim",
            "v_head_dim",
        ),
        axes=(AxisSpec("batch_size"),),
        dtype_fields=("attn_dtype", "kv_cache_dtype"),
        description="MLA core; context_lengths is a sorted context-length histogram normalized per request.",
    ),
    "routed_experts": TableSpec(
        name="routed_experts",
        key_fields=("dtype", "hidden_size", "inter_size", "expert_tokens"),
        axes=(),
        dtype_fields=("dtype",),
        description="One grouped SwiGLU branch for an exact per-expert assignment histogram; excludes router/shared/combine.",
    ),
    "mhc": TableSpec(
        name="mhc",
        key_fields=(
            "op_name",
            "architecture",
            "kernel_source",
            "hidden_size",
            "hc_mult",
            "num_sites",
            "sinkhorn_iters",
            "num_tokens",
        ),
        axes=(AxisSpec("num_tokens"),),
        dtype_fields=(),
        description="Measured compressed attention mHC pre/post TileLang module; residual BF16 with FP32 parameters.",
        metadata_fields=("source_hardware", "source_version", "source_framework"),
    ),
    "dsv4_attention": TableSpec(
        name="dsv4_attention",
        key_fields=(
            "layer_kind",
            "phase",
            "mla_dtype",
            "kv_cache_dtype",
            "kernel_source",
            "kernel_dtype",
            "model",
            "architecture",
            "num_heads",
            "tp_size",
            "batch_size",
            "input_seq_len",
            "context_len",
            "compress_ratio",
        ),
        axes=(
            AxisSpec("batch_size"),
            AxisSpec("input_seq_len", "linear", ATTENTION_SEQ_EXTRAPOLATION_RATIO),
            AxisSpec("context_len", "linear", ATTENTION_SEQ_EXTRAPOLATION_RATIO),
        ),
        dtype_fields=("mla_dtype", "kv_cache_dtype", "kernel_dtype"),
        description=(
            "Complete compressed attention CSA/HCA attention module including input projections, "
            "compressor/indexer, sparse attention and output projection."
        ),
        metadata_fields=("source_hardware", "source_version", "source_model"),
    ),
    "dsv4_megamoe": TableSpec(
        name="dsv4_megamoe",
        key_fields=(
            "dtype",
            "phase",
            "kernel_source",
            "kernel_dtype",
            "distribution",
            "pre_dispatch",
            "source_policy",
            "num_tokens",
            "hidden_size",
            "inter_size",
            "top_k",
            "num_experts",
            "num_fused_shared_experts",
            "tp_size",
            "ep_size",
        ),
        axes=(AxisSpec("num_tokens", "linear"),),
        dtype_fields=("dtype",),
        description="compressed attention routed MegaMoE incl pre-dispatch/dispatch/combine; rank-local tokens, gate/top-k and shared MLP excluded.",
        metadata_fields=(
            "num_max_tokens_per_rank",
            "effective_num_max_tokens_per_rank",
            "global_num_tokens",
            "routed_scaling_factor",
            "includes_gate_topk",
            "includes_buffer_init",
            "includes_routed_scale",
            "used_cuda_graph",
            "recorded_framework_version",
            "collection_version",
            "source_hardware",
        ),
    ),
    "gemm": TableSpec(
        name="gemm",
        key_fields=("dtype", "m", "n", "k"),
        # Outer axes are resolved first; ``m`` (tokens) varies most at runtime.
        axes=(AxisSpec("k"), AxisSpec("n"), AxisSpec("m")),
        dtype_fields=("dtype",),
        description="C[m,n] = A[m,k] x B[k,n]; dtype is the weight/storage dtype (e.g. fp16, fp8, int8_wo).",
    ),
    "grouped_gemm": TableSpec(
        name="grouped_gemm",
        key_fields=("dtype", "out_dtype", "batch_size", "m", "n", "k"),
        axes=(AxisSpec("batch_size"), AxisSpec("k"), AxisSpec("n"), AxisSpec("m")),
        dtype_fields=("dtype", "out_dtype"),
        description="One grouped/batched matrix kernel, independent weights per group; C[g,m,n] = A[g,m,k] B[g,k,n].",
    ),
    # Attention sequence axes may be extrapolated 64x past the measured range
    # (AIConfigurator's own tables stop at 16k prefill / 128k decode tokens
    # while agentic traces run to ~1M). Past the boundary the lookup keeps the
    # boundary row's measured efficiency and lets the analytical model carry
    # the O(s^2) growth, the rule AIConfigurator applies with no cap at all;
    # by 16k the kernels are compute-bound, so efficiency is flat in length.
    # for prefill
    "context_attention": TableSpec(
        name="context_attention",
        key_fields=(
            "attn_dtype",
            "kv_cache_dtype",
            "batch_size",
            "input_seq_len",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "window_size",
        ),
        axes=(
            AxisSpec("num_heads", "linear"),
            AxisSpec("batch_size"),
            AxisSpec(
                "input_seq_len",
                max_extrapolation_ratio=ATTENTION_SEQ_EXTRAPOLATION_RATIO,
            ),
        ),
        dtype_fields=("attn_dtype", "kv_cache_dtype"),
        description="Prefill attention core (QK^T, softmax, PV) for batch_size sequences padded to input_seq_len.",
    ),
    # for decode
    "generation_attention": TableSpec(
        name="generation_attention",
        key_fields=(
            "attn_dtype",
            "kv_cache_dtype",
            "batch_size",
            "context_len",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "window_size",
        ),
        axes=(
            AxisSpec("num_heads", "linear"),
            AxisSpec("batch_size"),
            AxisSpec(
                "context_len", max_extrapolation_ratio=ATTENTION_SEQ_EXTRAPOLATION_RATIO
            ),
        ),
        dtype_fields=("attn_dtype", "kv_cache_dtype"),
        description="Decode attention core for batch_size single-token queries against context_len cached tokens each.",
    ),
    "moe": TableSpec(
        name="moe",
        key_fields=(
            "dtype",
            "distribution",
            "num_tokens",
            "hidden_size",
            "inter_size",
            "top_k",
            "num_experts",
            "tp_size",
            "ep_size",
        ),
        axes=(AxisSpec("num_tokens"),),
        dtype_fields=("dtype",),
        description=(
            "Fused/grouped expert FFN time on one rank. num_tokens is the number of tokens "
            "entering the MoE layer (before top-k expansion) for the whole EP group; the row "
            "gives the time of one rank holding num_experts/ep_size experts sharded tp_size ways."
        ),
    ),
    "elementwise": TableSpec(
        name="elementwise",
        key_fields=("op_name", "dtype", "num_tokens", "hidden_size"),
        axes=(AxisSpec("hidden_size"), AxisSpec("num_tokens")),
        dtype_fields=("dtype",),
        description="Memory-bound point-wise kernels (rmsnorm, rope, residual_add, swiglu, data_reorder).",
    ),
    "collective": TableSpec(
        name="collective",
        key_fields=("dtype", "operation", "group_size", "nodes", "message_bytes"),
        axes=(AxisSpec("message_bytes"),),
        dtype_fields=("dtype",),
        description=(
            "Measured collective time (nccl-tests convention: message_bytes is the full buffer, "
            "time is one out-of-place iteration) for group_size ranks spread over nodes."
        ),
    ),
    "ep_all2all": TableSpec(
        name="ep_all2all",
        key_fields=(
            "dtype",
            "phase",
            "mode",
            "ep_size",
            "nodes",
            "hidden_size",
            "top_k",
            "num_experts",
            "num_tokens",
        ),
        # DeepEP cost is driven by num_tokens x top_k x hidden_size; the expert
        # count only sizes buffers, so it is an (outer) interpolation axis rather
        # than a discrete key and queries for unmeasured expert counts still hit
        # the nearest measured curve.
        axes=(
            AxisSpec("num_experts", max_extrapolation_ratio=1024.0),
            AxisSpec("top_k", "linear", max_extrapolation_ratio=8.0),
            AxisSpec("hidden_size"),
            AxisSpec("num_tokens"),
        ),
        dtype_fields=("dtype",),
        description=(
            "Measured expert-parallel dispatch or combine time (DeepEP and similar kernels) for one "
            "rank sending num_tokens local tokens, each routed to top_k of num_experts experts spread "
            "over ep_size ranks on nodes nodes. mode names the kernel family "
            "(deepep_high_throughput, deepep_low_latency, ...)."
        ),
    ),
}

INTEGER_FIELDS = frozenset(
    {
        "groups",
        "query_len",
        "kv_lora_rank",
        "rope_head_dim",
        "nope_head_dim",
        "v_head_dim",
        "m",
        "n",
        "k",
        "batch_size",
        "input_seq_len",
        "context_len",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "window_size",
        "num_tokens",
        "hidden_size",
        "inter_size",
        "top_k",
        "num_experts",
        "tp_size",
        "ep_size",
        "num_fused_shared_experts",
        "compress_ratio",
        "hc_mult",
        "num_sites",
        "sinkhorn_iters",
        "group_size",
        "nodes",
        "message_bytes",
    }
)

STRING_FIELDS = frozenset(
    field
    for spec in TABLE_SPECS.values()
    for field in spec.key_fields
    if field not in INTEGER_FIELDS
) | {"source_id"}


def table_spec(name: str) -> TableSpec:
    try:
        return TABLE_SPECS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown operator table {name!r}; known: {sorted(TABLE_SPECS)}"
        ) from exc
