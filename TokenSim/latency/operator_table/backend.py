"""Configure the operator backend and estimate scheduler-step latency."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any
from TokenSim.config.cache_config import (
    local_kv_heads,
    stage_layer_count,
    stage_layer_range,
)
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
    OperatorLookup,
)
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.workload import activation_bytes_for
from TokenSim.parallel import ParallelCommunicator
from .stats import OperatorStats, _MOE_DISTRIBUTION_CANDIDATES
from .composition import StepComposition
from .queries import OperatorQueries


class OperatorTableLatencyBackend(StepComposition, OperatorQueries, LatencyBackend):
    """Configure the operator backend and estimate scheduler-step latency."""

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
        pipeline_layer_counts: tuple[int, ...] | None = None,
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
        self.pipeline_layer_counts = pipeline_layer_counts
        if pipeline_layer_counts is not None:
            owned = stage_layer_range(
                model.num_layers,
                self.parallel_config.pipeline_parallel_size,
                self.rank_info.pp_rank,
                pipeline_layer_counts,
            )
            balanced = stage_layer_range(
                model.num_layers,
                self.parallel_config.pipeline_parallel_size,
                self.rank_info.pp_rank,
            )
            if owned != balanced and (model.is_moe or model.components):
                raise ConfigurationError(
                    "nonuniform operator-backend pipelines require a dense ordinary model"
                )
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
        self.trace_composition = None
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
        self.include_pp_transfer = self.include_step_overhead = True
        self.include_embedding_scale = self.rank_info.pp_rank == 0
        from .layers import layer_backends

        self._layer_backends = layer_backends(self)

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
        self.layers_local = (
            stage_layer_count(model.num_layers, pp, self.rank_info.pp_rank)
            if self.pipeline_layer_counts is None
            else len(
                stage_layer_range(
                    model.num_layers,
                    pp,
                    self.rank_info.pp_rank,
                    self.pipeline_layer_counts,
                )
            )
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
        self.moe_dtype = model.operator_dtype(
            "routed_experts", model.operator_dtype("routed_expert")
        )

    def estimate_step_latency(self, requests: list[Request]) -> float:
        """Latency of one step: decode tokens and prefill/recompute chunks may mix."""
        if not requests:
            return 0.0
        if self._layer_backends:
            steps = self.stats.step_count
            for child in self._layer_backends:
                child.moe_token_multiplier = self.moe_token_multiplier
            latency = sum(
                child.estimate_step_latency(requests) for child in self._layer_backends
            )
            self.stats.step_count = steps + 1
            return latency
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
                rows = req.logits_tokens if getattr(req, "emit_logits", True) else 0
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
            elif getattr(req, "emit_logits", True) and (
                is_decode_step(req) or kv_len >= req.context_len
            ):
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
            requests=sampling,
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
            "lookup_semantics": self.lookup.semantics if self.lookup else None,
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
