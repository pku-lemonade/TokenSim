"""Latency backend that composes per-operator table lookups into a step time.

For every scheduler step the backend decomposes the transformer layers owned
by this rank into GEMM, attention-core, MoE, elementwise and collective
operators, resolves each operator against the device's operator tables (exact
match, controlled interpolation, bounded extrapolation) and falls back to the
device family's analytical model when a table has no usable data. Every
estimate records where it came from, so results can report how much of the
simulated time rests on measurements versus formulas.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from TokenSim.comm.collectives import CollectiveModel
from TokenSim.config.cache_config import local_kv_heads, stage_layer_count
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.moe_execution import MoEExecutionConfig
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.utils import BoundedCache
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceSpec
from TokenSim.latency.base import (
    LatencyBackend,
    is_decode_step,
    is_decode_attention_step,
    step_kv_len,
    query_tokens,
)
from TokenSim.llm.llm_request import Request
from TokenSim.moe.placement import ExpertPlacement
from TokenSim.moe.routing import ExpertRouting
from TokenSim.moe.stats import MoEStats
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.coverage import MissingShapeReport
from TokenSim.operator_data.lookup import (
    FALLBACK_POLICIES,
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.workload import activation_bytes_for, context_attention_work
from TokenSim.parallel import ParallelCommunicator

logger = logging.getLogger(__name__)

COMPONENTS = ("gemm", "attention", "moe", "elementwise", "lm_head", "comm", "overhead")

_MOE_DISTRIBUTION_CANDIDATES = {
    "uniform": ("uniform", "balanced", "power_law_1.01"),
    "balanced": ("balanced", "uniform", "power_law_1.01"),
    "skew": ("power_law_1.2", "skew", "uniform", "balanced"),
    "hot": ("power_law_1.2", "hot", "uniform", "balanced"),
    "burst": ("power_law_1.2", "burst", "uniform", "balanced"),
}


@dataclass
class OperatorStats:
    """Aggregated provenance and time breakdown of every operator estimate.

    ``missing`` is the collection checklist of shapes no table answered. The
    worker shares the same report with its collective model, so communication
    misses (collective, ep_all2all) land in the same list as compute misses.
    """

    match_type_counts: Counter = field(default_factory=Counter)
    table_match_counts: dict[str, Counter] = field(default_factory=dict)
    component_seconds: Counter = field(default_factory=Counter)
    missing: MissingShapeReport = field(default_factory=MissingShapeReport)
    step_count: int = 0

    def record(self, table: str, match_type: str) -> None:
        self.match_type_counts[match_type] += 1
        self.table_match_counts.setdefault(table, Counter())[match_type] += 1

    def record_missing(self, error: MissingOperatorDataError) -> None:
        self.missing.record_error(error)

    def record_component(self, component: str, seconds: float) -> None:
        self.component_seconds[component] += seconds

    def aggregate(self, other: "OperatorStats") -> "OperatorStats":
        result = OperatorStats(missing=self.missing.merge(other.missing))
        result.match_type_counts = self.match_type_counts + other.match_type_counts
        for table, counter in list(self.table_match_counts.items()) + list(
            other.table_match_counts.items()
        ):
            result.table_match_counts[table] = (
                result.table_match_counts.get(table, Counter()) + counter
            )
        result.component_seconds = self.component_seconds + other.component_seconds
        result.step_count = self.step_count + other.step_count
        return result

    def as_dict(self) -> dict[str, Any]:
        total = sum(self.match_type_counts.values())
        return {
            "operator_query_count": total,
            "operator_match_type_counts": dict(self.match_type_counts),
            "operator_table_match_counts": {
                table: dict(counter)
                for table, counter in sorted(self.table_match_counts.items())
            },
            "operator_component_seconds": {
                k: self.component_seconds.get(k, 0.0) for k in COMPONENTS
            },
            "operator_missing_shape_count": self.missing.shape_count,
            "operator_missing_shape_groups": self.missing.group_count,
            "operator_missing_per_table": self.missing.per_table(),
            "operator_step_count": self.step_count,
        }

    def missing_shape_records(self) -> list[dict[str, Any]]:
        """Missing shapes grouped by table and discrete key, most frequent first."""
        return self.missing.records()


@dataclass(frozen=True)
class OperatorEstimate:
    latency_us: float
    match_type: str
    source_id: str
    workload_distribution: str | None = None


class OperatorTableLatencyBackend(LatencyBackend):
    _CACHE_LIMIT = 262144

    def __init__(
        self,
        *,
        device: DeviceSpec,
        model: ModelSpec,
        parallel_config: ParallelConfig | None = None,
        rank_info: ParallelRankInfo | None = None,
        communicator: ParallelCommunicator | None = None,
        expert_placement: ExpertPlacement | None = None,
        package: OperatorDataPackage | None = None,
        fallback: str = "table_first",
        lookup_policy: LookupPolicy | None = None,
        decode_context_bucket: int = 128,
        random_seed: int = 0,
        missing_report: MissingShapeReport | None = None,
        moe_execution=None,
        attention_backend: str = "generic",
        mhc_backend: str = "generic",
        expected_decode_query_len: int = 1,
        max_num_seqs: int = 64,
    ) -> None:
        if fallback not in FALLBACK_POLICIES:
            raise ConfigurationError(
                f"fallback must be one of {FALLBACK_POLICIES}, got {fallback!r}"
            )
        if (
            fallback != "analytical_only"
            and package is None
            and fallback == "table_only"
        ):
            raise ConfigurationError(
                f"latency backend 'table_only' requires operator data for device {device.device_id!r}"
            )
        self.device = device
        self.model = model
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.rank_info = rank_info or ParallelRankInfo()
        self.communicator = communicator
        if expert_placement is None and model.is_moe:
            from TokenSim.moe.placement import build_expert_placement

            expert_placement = build_expert_placement(
                model.moe, self.parallel_config, total_layers=model.num_layers
            )
        self.expert_placement = expert_placement
        self.package = package if fallback != "analytical_only" else None
        self.fallback = fallback
        self.analytical = analytical_model_for(device.family)
        # Out-of-range table queries keep the boundary row's measured efficiency
        # and follow the analytical model's growth (see LookupPolicy).
        self.lookup = (
            OperatorLookup(
                self.package,
                lookup_policy,
                analytical_scaler=self._analytical_reference_us,
            )
            if self.package is not None
            else None
        )
        self.decode_context_bucket = max(1, int(decode_context_bucket))
        self.stats = OperatorStats(missing=missing_report or MissingShapeReport())
        self.target_logits_counts = Counter()
        self.main_forward_counts = Counter()
        self.moe_stats = MoEStats(
            effective_moe_config=model.moe.to_dict(),
            expert_placement=expert_placement.to_dict() if expert_placement else None,
        )
        self.expert_routing = ExpertRouting(
            model.moe,
            seed=random_seed,
            rank_to_experts=(
                expert_placement.rank_to_experts if expert_placement else None
            ),
        )
        self._expert_to_rank: dict[int, int] | None = None
        self._rank_load_cache = BoundedCache(4096)
        self._cache = BoundedCache(self._CACHE_LIMIT)
        self._calibrations = {
            table: Calibration.from_mapping(values)
            for table, values in (
                self.package.meta.calibration.items() if self.package else []
            )
        }
        self._init_dimensions()
        if attention_backend not in ("generic", "module"):
            raise ConfigurationError(
                f"unsupported compressed attention backend: {attention_backend}"
            )
        self.attention_backend = attention_backend
        self.mtp_query_width = max(1, int(expected_decode_query_len))
        self.max_num_seqs = max_num_seqs
        self.moe_step = None
        self.residual_output = None
        if model.components and self.is_last_stage:
            from TokenSim.latency.output import ResidualOutputEstimator

            self.residual_output = ResidualOutputEstimator(self, scope="target")
        self.native_attention = None
        if attention_backend == "module":
            from TokenSim.latency.attention import (
                CompressedAttentionEstimator,
            )

            self.native_attention = CompressedAttentionEstimator(self)
        if mhc_backend not in ("generic", "module"):
            raise ConfigurationError(
                f"unsupported residual mixing backend: {mhc_backend}"
            )
        self.mhc_backend = mhc_backend
        self.native_mhc = None
        if mhc_backend == "module":
            from TokenSim.latency.residual import ResidualMixingEstimator

            self.native_mhc = ResidualMixingEstimator(self)
        self.moe_execution = MoEExecutionConfig.from_value(moe_execution)
        self.native_moe = None
        if self.moe_execution.native:
            from TokenSim.latency.moe import MegaMoEEstimator

            self.native_moe = MegaMoEEstimator(
                self,
                self.moe_execution,
                expected_query_width=expected_decode_query_len,
                max_num_seqs=max_num_seqs,
            )
        self.native_mtp = None
        if model.components and self.mtp_query_width > 1:
            from TokenSim.latency.speculative import SpeculativeEstimator

            self.native_mtp = SpeculativeEstimator(self, max_num_seqs=max_num_seqs)

    # -- dimensions -------------------------------------------------------------

    def _init_dimensions(self) -> None:
        model = self.model
        tp = self.parallel_config.tensor_parallel_size
        pp = self.parallel_config.pipeline_parallel_size
        self.tp = tp
        self.heads_local = math.ceil(model.num_attention_heads / tp)
        self.kv_heads_local = local_kv_heads(model.num_key_value_heads, tp)
        self.q_dim_local = self.heads_local * model.head_dim
        self.kv_dim_local = (
            model.kv_dim // tp
            if model.kv_cache_dim is not None
            else self.kv_heads_local * model.head_dim
        )
        self.inter_local = math.ceil(model.intermediate_size / tp)
        self.vocab_local = math.ceil(model.vocab_size / tp)
        self.layers_local = stage_layer_count(
            model.num_layers, pp, self.rank_info.pp_rank
        )
        self.is_last_stage = self.rank_info.pp_rank == pp - 1
        self.moe_layers_local = (
            len(self.expert_placement.moe_layers_for_pp_rank(self.rank_info.pp_rank))
            if model.is_moe and self.expert_placement is not None
            else 0
        )
        self.dense_layers_local = max(0, self.layers_local - self.moe_layers_local)
        from TokenSim.latency.projection import attention_projections

        self.native_attention_projections = attention_projections(model, tp)
        self.activation_bytes = activation_bytes_for(model.activation_dtype)
        ep_enabled = (
            self.parallel_config.enable_expert_parallel
            and self.expert_placement is not None
            and self.expert_placement.ep_rank_count > 1
        )
        self.ep_enabled = bool(ep_enabled)
        self.moe_tp = 1 if ep_enabled else tp
        self.moe_ep = self.expert_placement.ep_rank_count if ep_enabled else 1
        # Every DP replica inside this rank's expert-parallel group dispatches
        # into the same expert ranks; assume lockstep steps, so the group sees
        # ``replicas_per_group`` times this rank's tokens (1 with per-DP EP).
        self.moe_token_multiplier = (
            self.parallel_config.expert_parallel_replicas_per_group if ep_enabled else 1
        )
        routing = model.moe.routing.distribution if model.moe.routing else "uniform"
        distribution = (
            model.operator_modules.get("moe", {}).get("key", {}).get("distribution")
        )
        self.moe_distributions = (
            (distribution,)
            if distribution
            else _MOE_DISTRIBUTION_CANDIDATES.get(routing, ("uniform", "balanced"))
        )
        self.moe_dtype = model.operator_dtype("routed_expert")

    # -- public API -------------------------------------------------------------

    def estimate_step_latency(self, requests: list[Request]) -> float:
        """Latency of one step: decode tokens and prefill/recompute chunks may mix."""
        if not requests:
            return 0.0
        self.stats.step_count += 1
        tokens = 0
        sampling = 0
        decode_logits_rows = 0
        prefill_logits_rows = 0
        context_groups: Counter = Counter()
        decode_buckets: Counter = Counter()
        for req in requests:
            q_len = query_tokens(req)
            kv_len = step_kv_len(req)
            tokens += q_len
            decode_attention = (
                is_decode_attention_step(req, self.mtp_query_width)
                if self.native_attention
                else is_decode_step(req)
            )
            if decode_attention:
                decode_buckets[(self._bucket(kv_len), q_len)] += 1
            else:
                # Chunk of a context build: q_len new tokens attending to the
                # kv_len tokens present after this step (cached prefix, earlier
                # chunks and the chunk itself).
                context_groups[(q_len, kv_len)] += 1
            if self.model.components:
                # Logits ownership follows proposals, not attention metadata.
                rows = req.logits_tokens
                sampling += rows
                if req.is_context_build:
                    prefill_logits_rows += rows
                    self.target_logits_counts["context_decode_metadata_rows"] += (
                        q_len if decode_attention else 0
                    )
                    self.target_logits_counts[
                        "placeholder_rows"
                    ] += req.scheduled_placeholder_tokens
                    self.target_logits_counts["context_graph_eligible_requests"] += int(
                        req.decode_graph_eligible
                    )
                else:
                    decode_logits_rows += rows
            elif is_decode_step(req) or kv_len >= req.context_len:
                sampling += 1
        if self.model.components and self.is_last_stage:
            self.target_logits_counts["steps"] += 1
            self.target_logits_counts["decode_rows"] += decode_logits_rows
            self.target_logits_counts["prefill_rows"] += prefill_logits_rows
            self.target_logits_counts["rows"] += sampling
        if self.native_attention:
            forward_rows = self._target_forward_rows(tokens, requests)
            attention_us = self.native_attention.estimate_batch(
                {
                    (q_len, max(0, kv_len - q_len)): count
                    for (q_len, kv_len), count in context_groups.items()
                },
                decode_buckets,
                projection_rows=forward_rows,
            )
        else:
            attention_us = sum(
                self._attention_prefill(count, q_len, kv_len)
                for (q_len, kv_len), count in context_groups.items()
            ) + sum(
                self._attention_decode(count, context, query_width)
                for (context, query_width), count in decode_buckets.items()
            )
        return self._compose(
            tokens=tokens,
            requests=max(1, sampling),
            attention_us=attention_us,
            requests_list=requests,
        )

    def describe(self) -> dict[str, Any]:
        description = {
            "backend": "operator_table",
            "fallback": self.fallback,
            "device_id": self.device.device_id,
            "device_family": self.device.family,
            "model_id": self.model.model_id,
            "dataset_version": self.package.dataset_version if self.package else None,
            "operator_backend": self.package.backend if self.package else "analytical",
            "tables": (
                {name: len(rows) for name, rows in self.package.tables.items()}
                if self.package
                else {}
            ),
            "analytical_model": type(self.analytical).__name__,
            "decode_context_bucket": self.decode_context_bucket,
            "missing_shape_scope": "cumulative collection checklist; estimate counters reset after warmup",
            "moe_cost_dtype": self.moe_dtype,
            "moe_cost_distribution_candidates": list(self.moe_distributions),
            "moe_route_statistics": "whole-step synthetic routes; EP-interleaved rotating remainder; explicit histograms retained",
            "moe_histogram_changes_latency": not self.native_moe
            and not all(d.startswith("power_law") for d in self.moe_distributions),
            "moe_kernel_scope": (
                "fused module incl dispatch/combine + separately estimated shared experts; gate/top-k outside"
                if self.native_moe
                else "routed compute table and explicit EP communication"
            ),
        }
        if self.native_attention_projections:
            description["attention_projections"] = [
                {
                    "name": p.name,
                    "table": p.table,
                    "dtype": p.dtype,
                    "n": p.n,
                    "k": p.k,
                    "groups": p.groups,
                }
                for p in self.native_attention_projections
            ]
            description["attention_cost_scope"] = (
                "module table includes projections, compressor/indexer, sparse attention and output projection"
                if self.native_attention
                else "projection plan and independently priced attention core"
            )
        if self.model.components:
            description["main_forward"] = {
                "full_rows": "eligible DP decode descriptor, otherwise logical forward tensor",
                "sp_rows": "ceil(full/TP) for PP1/EP/TP>1 sequence-parallel residual mixing/norm/router",
                "attention_rows": "full tensor after gather, sliced to positions; actual sparse/KV queries unchanged",
                "padding": "compute-only rows, never request progress or selected logits",
            }
        if self.residual_output:
            description["target_output"] = {
                "residual_output": self.residual_output.describe(),
                "forward_rows": "full target batch, including eligible synchronized decode capture padding",
                "final_norm_rows": "same full forward rows, after SP gather",
                "lm_head_rows": "selected decode queries and one bonus row per prefill request",
            }
        for name, module in (
            ("attention_module", self.native_attention),
            ("residual_mixing_module", self.native_mhc),
            ("moe_module", self.native_moe),
            ("speculative", self.native_mtp),
        ):
            if module:
                description[name] = module.describe()
        return description

    def set_moe_step(self, step):
        self.moe_step = step
        if self.native_moe:
            self.native_moe.set_step(step)
        if self.native_mtp:
            self.native_mtp.set_step(step)

    def reset_profile_stats(self):
        # Retain lookup caches and the shared compute/communication collection
        # checklist, but remove warmup estimates from measured time breakdown.
        self.stats = OperatorStats(missing=self.stats.missing)
        self.target_logits_counts.clear()
        self.main_forward_counts.clear()
        if self.residual_output:
            self.residual_output.reset_stats()
        if self.native_mtp:
            self.native_mtp.reset_stats()
        if self.native_moe:
            self.native_moe.reset_stats()
        if self.native_attention:
            self.native_attention.reset_stats()
        if self.native_mhc:
            self.native_mhc.reset_stats()

    # -- step composition -------------------------------------------------------

    def _bucket(self, context_len: int) -> int:
        step = self.decode_context_bucket
        return int(math.ceil(context_len / step) * step)

    def _target_forward_rows(self, tokens, requests_list):
        if not self.model.components or self.moe_step is None and not requests_list:
            return tokens
        from TokenSim.llm.forward import ForwardPass, RankShape, uniform_decode_rows

        step = self.moe_step
        if step is None:
            widths = {query_tokens(req) for req in requests_list}
            context = any(not req.decode_graph_eligible for req in requests_list)
            step = ForwardPass(
                {
                    self.rank_info.dp_rank: RankShape(
                        tokens,
                        (
                            next(iter(widths))
                            if len(widths) == 1 and not context
                            else None
                        ),
                        context,
                        len(requests_list),
                    )
                }
            )
        padded = uniform_decode_rows(step, self.mtp_query_width, self.max_num_seqs)
        return padded if padded is not None else tokens

    def _compose(
        self,
        *,
        tokens: int,
        requests: int,
        attention_us: float,
        requests_list: list[Request],
    ) -> float:
        """Sum every operator of the layers on this rank for a step of ``tokens`` tokens.

        ``requests`` is the number of target logits rows (one per prefill
        request, and one per decode query under compressed attention MTP); ``attention_us`` is
        the attention core already resolved per
        shape by the caller.
        """
        model = self.model
        h = model.hidden_size
        T = tokens
        stats = self.stats
        forward_rows = self._target_forward_rows(T, requests_list)

        # Attention block shared by dense and MoE layers.
        if self.native_attention:
            # The complete compressed attention module table already contains all
            # four projections, compressor/indexer and output projection.
            attn_gemm_us = 0.0
        elif self.native_attention_projections:
            attn_gemm_us = sum(
                self.query(p.table, p.key(forward_rows)).latency_us
                for p in self.native_attention_projections
            )
        else:
            attn_gemm_us = self.gemm(
                T, self.q_dim_local + 2 * self.kv_dim_local, h
            ) + self.gemm(T, h, self.q_dim_local)
        # mHC runs on SP shards before the attention all-gather. Match the
        # serving predicate; TP1 DEP8 keeps its full row count.
        mhc_sp = bool(
            model.components
            and self.ep_enabled
            and self.parallel_config.pipeline_parallel_size == 1
            and self.parallel_config.sequence_parallel_moe(self.native_moe is not None)
        )
        norm_rows = math.ceil(forward_rows / self.tp) if mhc_sp else forward_rows
        router_rows = norm_rows if model.components else T
        if model.components:
            counts = self.main_forward_counts
            counts["steps"] += 1
            counts["logical_rows"] += T
            counts["full_rows"] += forward_rows
            counts["capture_padding_rows"] += forward_rows - T
            counts["sp_rows"] += norm_rows
            counts["sp_tail_rows"] += (
                norm_rows * self.tp - forward_rows if mhc_sp else 0
            )
            counts["sp_steps"] += int(mhc_sp)
            counts["graph_padded_steps"] += int(forward_rows > T)
            counts["attention_projection_layer_rows"] += (
                forward_rows * self.layers_local
            )
            counts["mhc_layer_rows"] += (
                norm_rows * self.layers_local if self.native_mhc else 0
            )
            counts["norm_layer_rows"] += norm_rows * (
                self.layers_local + self.moe_layers_local
            )
            counts["router_layer_rows"] += router_rows * self.moe_layers_local
        attn_elem_us = self.elementwise("rmsnorm", norm_rows, h)
        if not self.native_mhc:
            attn_elem_us += self.elementwise("residual_add", forward_rows, h)
        mhc_us = self.native_mhc.estimate(norm_rows) if self.native_mhc else 0.0
        if not self.native_attention:
            attn_elem_us += self.elementwise(
                "rope", T, self.q_dim_local + self.kv_dim_local
            )
        dense_ffn_gemm_us = 0.0
        dense_ffn_elem_us = 0.0
        if self.dense_layers_local:
            dense_ffn_gemm_us = self.gemm(
                T, model.ffn_projection_count * self.inter_local, h
            ) + self.gemm(T, h, self.inter_local)
            dense_ffn_elem_us = self.elementwise("rmsnorm", T, h) + self.elementwise(
                "residual_add", T, h
            )
            if model.gated:
                dense_ffn_elem_us += self.elementwise("swiglu", T, self.inter_local)

        moe_us = 0.0
        router_us = 0.0
        moe_elem_us = 0.0
        moe_comm_bytes = 0
        if self.moe_layers_local:
            router_us = self.gemm(
                router_rows,
                model.moe.num_experts,
                h,
                dtype=model.operator_dtype("router"),
            )
            moe_elem_us = self.elementwise("rmsnorm", norm_rows, h)
            if not self.native_mhc:
                # Native post already reconstructs the weighted residual.
                # The unfused collector pre does not own the output RMSNorm.
                moe_elem_us += self.elementwise("residual_add", forward_rows, h)
            moe_us = self._moe_layers(T, requests_list)
            if self.ep_enabled and not self.native_moe:
                moe_comm_bytes = int(
                    T * model.moe.num_experts_per_tok * h * self.activation_bytes
                )

        layer_gemm_us = (
            self.layers_local * attn_gemm_us
            + self.dense_layers_local * dense_ffn_gemm_us
            + self.moe_layers_local * router_us
        )
        layer_elem_us = (
            self.layers_local * attn_elem_us
            + self.dense_layers_local * dense_ffn_elem_us
            + self.moe_layers_local * moe_elem_us
            + self.layers_local * mhc_us
        )
        layer_attention_us = (
            attention_us if self.native_attention else self.layers_local * attention_us
        )

        lm_head_us = 0.0
        if self.is_last_stage:
            lm_head_us = self.gemm(
                max(1, requests),
                self.vocab_local,
                h,
                dtype=model.operator_dtype("lm_head"),
            )
            if self.residual_output:
                output_rows = forward_rows
                # The target reduces the hyper-residual and normalizes every
                # forward row before logits selection. SP has already gathered.
                lm_head_us += self.residual_output.estimate(output_rows)
                lm_head_us += self.elementwise("rmsnorm", output_rows, h)
                self.target_logits_counts["forward_rows"] += T
                self.target_logits_counts["output_forward_rows"] += output_rows
                self.target_logits_counts["hc_head_rows"] += output_rows
                self.target_logits_counts["final_norm_rows"] += output_rows
            else:
                lm_head_us += self.elementwise("rmsnorm", max(1, requests), h)

        overhead_us = self.device.step_overhead_us.value

        draft = self.native_mtp.estimate(requests_list) if self.native_mtp else {}
        layer_gemm_us += draft.get("gemm", 0)
        layer_attention_us += draft.get("attention", 0)
        layer_elem_us += draft.get("elementwise", 0)
        moe_us += draft.get("moe", 0)
        lm_head_us += draft.get("lm_head", 0)

        stats.record_component("gemm", layer_gemm_us * 1e-6)
        stats.record_component("attention", layer_attention_us * 1e-6)
        stats.record_component("elementwise", layer_elem_us * 1e-6)
        stats.record_component("moe", moe_us * 1e-6)
        stats.record_component("lm_head", lm_head_us * 1e-6)
        stats.record_component("overhead", overhead_us * 1e-6)

        compute_s = (
            layer_gemm_us
            + layer_attention_us
            + layer_elem_us
            + moe_us
            + lm_head_us
            + overhead_us
        ) * 1e-6
        comm_s = (
            self._communication(forward_rows, moe_comm_bytes)
            + draft.get("comm", 0) * 1e-6
        )
        stats.record_component("comm", comm_s)
        return compute_s + comm_s

    def _communication(self, tokens: int, moe_comm_bytes: int) -> float:
        if self.communicator is None:
            return 0.0
        hidden_bytes = int(tokens * self.model.hidden_size * self.activation_bytes)
        latency = 0.0
        if self.tp > 1:
            self.communicator.record_tp_shard()
            # Row-parallel output projection and FFN down projection each
            # synchronize the hidden state: one all-reduce after attention in
            # every layer, one after the FFN in dense layers and in MoE layers
            # sharded with pure TP. Expert-parallel MoE layers replace the
            # FFN all-reduce with dispatch/combine all-to-all below.
            allreduce_count = self.layers_local + self.dense_layers_local
            if not self.ep_enabled:
                allreduce_count += self.moe_layers_local
            latency += self.communicator.estimate_tp_collective(
                hidden_bytes, count=allreduce_count
            )
        if self.ep_enabled and moe_comm_bytes > 0 and self.moe_layers_local:
            # One dispatch + one combine per MoE layer; measured DeepEP rows are
            # keyed by this rank's token count and the expert configuration.
            latency += self.communicator.estimate_ep_all2all(
                moe_comm_bytes,
                count=self.moe_layers_local,
                num_tokens=tokens,
                hidden_size=self.model.hidden_size,
                top_k=self.model.moe.num_experts_per_tok,
                num_experts=self.model.moe.num_experts,
                activation_bytes=self.activation_bytes,
            )
        if self.parallel_config.pipeline_parallel_size > 1:
            latency += self.communicator.estimate_pp_stage_transfer(hidden_bytes)
        return latency

    # -- operator queries -------------------------------------------------------

    def gemm(self, m: int, n: int, k: int, *, dtype: str | None = None) -> float:
        if m <= 0 or n <= 0 or k <= 0:
            return 0.0
        return self.query(
            "gemm",
            {"dtype": dtype or self.model.dtype, "m": int(m), "n": int(n), "k": int(k)},
        ).latency_us

    def elementwise(self, op_name: str, tokens: int, hidden: int) -> float:
        if tokens <= 0 or hidden <= 0:
            return 0.0
        return self.query(
            "elementwise",
            {
                "op_name": op_name,
                "dtype": self.model.activation_dtype,
                "num_tokens": int(tokens),
                "hidden_size": int(hidden),
            },
        ).latency_us

    def _attention_prefill(self, batch: int, q_len: int, kv_len: int) -> float:
        model = self.model
        key = {
            "attn_dtype": model.activation_dtype,
            "kv_cache_dtype": model.kv_cache_dtype,
            "batch_size": int(batch),
            "input_seq_len": int(q_len),
            "num_heads": self.heads_local,
            "num_kv_heads": self.kv_heads_local,
            "head_dim": model.head_dim,
            "window_size": model.sliding_window,
        }
        latency = self.query("context_attention", key).latency_us
        if kv_len > q_len:
            # Prefix-cache hit: queries attend to cached keys as well. Scale by
            # the analytical work ratio because tables assume kv_len == q_len.
            base = context_attention_work(
                batch,
                q_len,
                self.heads_local,
                self.kv_heads_local,
                model.head_dim,
                model.activation_dtype,
                model.kv_cache_dtype,
                model.sliding_window,
            )
            full_keys = float(
                min(kv_len, model.sliding_window) if model.sliding_window else kv_len
            )
            # queries q_len each attend to (kv_len - q_len) cached keys + causal part of their own block
            full_flops = (
                4.0
                * batch
                * q_len
                * ((kv_len - q_len) + 0.5 * q_len)
                * self.q_dim_local
            )
            if model.sliding_window:
                full_flops = min(
                    full_flops, 4.0 * batch * q_len * full_keys * self.q_dim_local
                )
            if base.flops > 0:
                latency *= max(1.0, full_flops / base.flops)
        return latency

    def _attention_decode(
        self, batch: int, context_len: int, query_width: int = 1
    ) -> float:
        model = self.model
        key = {
            "attn_dtype": model.activation_dtype,
            "kv_cache_dtype": model.kv_cache_dtype,
            "batch_size": int(batch),
            "context_len": int(context_len),
            "num_heads": self.heads_local,
            "num_kv_heads": self.kv_heads_local,
            "head_dim": model.head_dim,
            "window_size": model.sliding_window,
        }
        return self.query("generation_attention", key).latency_us

    def _moe_layers(self, tokens: int, requests: list[Request]) -> float:
        """Expert compute for all MoE layers on this rank, including straggler time."""
        model = self.model
        if self.expert_placement is None:
            return 0.0
        histogram = self.expert_routing.histogram_for_step(requests)
        per_rank_load = self._per_rank_load(histogram)
        if self.native_moe:
            total_us, routed_us, shared_us = self.native_moe.estimate(tokens)
            self.moe_stats.record_step(
                histogram=histogram,
                per_rank_load=per_rank_load,
                compute_latency=total_us * 1e-6,
                all2all_latency=0,
                straggler_latency=0,
                all2all_event_count=1,
            )
            return total_us
        imbalance = 1.0
        if per_rank_load:
            mean_load = sum(per_rank_load.values()) / max(
                1, self.expert_placement.ep_rank_count
            )
            max_load = max(per_rank_load.values())
            if mean_load > 0 and max_load > mean_load:
                imbalance = max_load / mean_load
        group_tokens = tokens * self.moe_token_multiplier
        base_key = {
            "dtype": self.moe_dtype,
            "hidden_size": model.hidden_size,
            "inter_size": model.moe_intermediate_size,
            "top_k": model.moe.num_experts_per_tok,
            "num_experts": model.moe.num_experts,
            "tp_size": self.moe_tp,
            "ep_size": self.moe_ep,
        }
        balanced = self.moe_query(
            {**base_key, "num_tokens": max(1, int(round(group_tokens)))}
        )
        compute_s = balanced.latency_us * self.moe_layers_local * 1e-6
        straggler_s = 0.0
        # A power-law measurement already includes its distribution's slow
        # ranks. compressed attention deliberately uses a fixed power_law_1.01 cost assumption;
        # its synthetic/trace histograms must never silently change that cost.
        measured_skew = bool(
            balanced.workload_distribution
            and balanced.workload_distribution.startswith("power_law")
        )
        if imbalance > 1.0 and not measured_skew:
            # The step waits for the busiest expert rank. Its load equals a
            # balanced layer fed with ``imbalance`` times as many tokens, which
            # keeps memory-bound small batches from being over-scaled.
            busiest = self.moe_query(
                {**base_key, "num_tokens": max(1, int(round(group_tokens * imbalance)))}
            )
            straggler_s = (
                max(0.0, busiest.latency_us - balanced.latency_us)
                * self.moe_layers_local
                * 1e-6
            )
        all2all_s = 0.0  # recorded by the communicator separately
        self.moe_stats.record_step(
            histogram=histogram,
            per_rank_load=per_rank_load,
            compute_latency=compute_s,
            all2all_latency=all2all_s,
            straggler_latency=straggler_s,
            all2all_event_count=1 if (self.ep_enabled and self.moe_layers_local) else 0,
        )
        return (compute_s + straggler_s) * 1e6

    def moe_query(self, base_key: Mapping[str, Any]) -> OperatorEstimate:
        last_error: MissingOperatorDataError | None = None
        if self.lookup is not None and self.fallback != "analytical_only":
            for distribution in self.moe_distributions:
                key = {**base_key, "distribution": distribution}
                cached = self._cache.get(("moe", tuple(sorted(key.items()))))
                if cached is not None:
                    self.stats.record("moe", cached.match_type)
                    return cached
                try:
                    result = self.lookup.lookup("moe", key)
                except MissingOperatorDataError as exc:
                    last_error = exc
                    continue
                estimate = OperatorEstimate(
                    result.latency_us,
                    result.match_type,
                    result.source_id,
                    workload_distribution=distribution,
                )
                self._remember(("moe", tuple(sorted(key.items()))), estimate)
                self.stats.record("moe", estimate.match_type)
                return estimate
        key = {**base_key, "distribution": self.moe_distributions[0]}
        if last_error is None:
            last_error = MissingOperatorDataError(
                "moe", key, "no moe table", kind="table_absent"
            )
        if self.fallback == "table_only":
            raise last_error
        self.stats.record_missing(last_error)
        return self._analytical("moe", key, gated=self.model.gated)

    def query(self, table: str, key: Mapping[str, Any]) -> OperatorEstimate:
        identity = (table, tuple(sorted(key.items())))
        cached = self._cache.get(identity)
        if cached is not None:
            self.stats.record(table, cached.match_type)
            return cached
        estimate: OperatorEstimate | None = None
        if self.lookup is not None and self.fallback != "analytical_only":
            try:
                result = self.lookup.lookup(table, key)
                estimate = OperatorEstimate(
                    result.latency_us, result.match_type, result.source_id
                )
            except MissingOperatorDataError as exc:
                if self.fallback == "table_only":
                    raise
                self.stats.record_missing(exc)
        if estimate is None:
            estimate = self._analytical(table, key, gated=self.model.gated)
        self._remember(identity, estimate)
        self.stats.record(table, estimate.match_type)
        return estimate

    def _analytical(
        self, table: str, key: Mapping[str, Any], **context: Any
    ) -> OperatorEstimate:
        calibration = self._calibrations.get(table) or self._calibrations.get("default")
        result = self.analytical.estimate(
            table, key, self.device, calibration, **context
        )
        return OperatorEstimate(result.latency_us, "analytical", result.source_id)

    def _analytical_reference_us(
        self, table: str, key: Mapping[str, Any]
    ) -> float | None:
        """Uncalibrated analytical latency used as the growth reference for extrapolation."""
        try:
            return self.analytical.estimate(
                table, key, self.device, Calibration(), gated=self.model.gated
            ).latency_us
        except Exception:
            return None

    def _remember(self, identity: tuple, estimate: OperatorEstimate) -> None:
        self._cache[identity] = estimate

    def _per_rank_load(self, histogram: Mapping[int, int]) -> dict[int, int]:
        if self.expert_placement is None:
            return {}
        # Routing reuses its cached read-only histograms across steps.
        cached = self._rank_load_cache.get(id(histogram))
        if cached is not None and cached[0] is histogram:
            return cached[1]
        if self._expert_to_rank is None:
            mapping: dict[int, int] = {}
            for ep_rank, experts in self.expert_placement.rank_to_experts.items():
                for expert_id in experts:
                    mapping[expert_id] = ep_rank
            self._expert_to_rank = mapping
        per_rank: dict[int, int] = {}
        for expert_id, count in histogram.items():
            ep_rank = self._expert_to_rank.get(expert_id)
            if ep_rank is None:
                continue
            per_rank[ep_rank] = per_rank.get(ep_rank, 0) + count
        # Holding the histogram keeps its id from being reused.
        self._rank_load_cache[id(histogram)] = (histogram, per_rank)
        return per_rank


def collective_model_for(
    device: DeviceSpec,
    placement,
    links,
    package: OperatorDataPackage | None,
    *,
    fallback: str = "table_first",
    missing_report: MissingShapeReport | None = None,
) -> CollectiveModel:
    """Build the collective model a worker on ``device`` should use.

    ``fallback`` and ``missing_report`` are the worker's operator-table policy
    and checklist, so communication misses are reported (or rejected under
    ``table_only``) exactly like compute misses.
    """
    has_comm_tables = package is not None and bool(
        {"collective", "ep_all2all"} & set(package.tables)
    )
    measured = OperatorLookup(package) if has_comm_tables else None
    launch_us = device.analytical.get("collective_launch_us")
    return CollectiveModel(
        placement.topology,
        links,
        family=device.family,
        measured_lookup=measured,
        launch_us=float(launch_us) if launch_us is not None else None,
        fallback=fallback,
        missing_report=missing_report,
    )
