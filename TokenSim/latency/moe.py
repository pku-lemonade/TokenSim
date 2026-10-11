"""Fused MegaMoE geometry, measured routed scope and shared MLP cost.

The collector's phase is a buffer/token-grid regime, not a kernel argument.
Unequal/partially padded passes normalize source-derived physical work to a
matching uniform module sample. The composition remains an explicit estimate.
"""

from collections import Counter
from dataclasses import dataclass
import math
from functools import lru_cache
from pathlib import Path

from TokenSim.utils import BoundedCache
from TokenSim.errors import ConfigurationError
from TokenSim.llm.forward import (
    ForwardPass,
    RankShape,
    sp_valid_rows,
    uniform_decode_rows,
)
from TokenSim.operator_data.lookup import LookupPolicy, OperatorLookup
from TokenSim.hardware.links import LinkCatalog
from TokenSim.config.paths import default_data_root
from TokenSim.latency.moe_work import MegaMoEWorkModel


@lru_cache(maxsize=4)
def _default_links(root):
    return LinkCatalog.load(Path(root) / "topologies/links.yaml")


@dataclass(frozen=True)
class GroupRows:
    inputs: dict[int, int]
    routed: dict[int, int]
    own_rank: int
    graph: bool
    step: ForwardPass


@dataclass(frozen=True)
class ModuleEstimate:
    latency_us: float
    match: str
    source_id: str
    regime: str


class MegaMoEEstimator:
    def __init__(self, backend, config, *, expected_query_width=1, max_num_seqs=64):
        self.backend, self.config = backend, config
        self.model, self.parallel = backend.model, backend.parallel_config
        binding = self.model.operator_modules.get("moe", {})
        self.table = config.table or binding.get("table")
        if not self.table:
            raise ConfigurationError(
                "measured MegaMoE requires an operator_modules.moe table binding"
            )
        self.expected_query_width, self.max_num_seqs = (
            expected_query_width,
            max_num_seqs,
        )
        if not self.model.moe.enabled or not backend.ep_enabled:
            raise ConfigurationError(
                "MegaMoE requires routed experts and expert parallel"
            )
        if self.parallel.pipeline_parallel_size != 1:
            raise ConfigurationError(
                "MegaMoE requires a complete PP1 expert group and matching measured table"
            )
        if (
            backend.fallback == "analytical_only"
            or backend.package is None
            or not backend.package.tables.get(self.table)
        ):
            raise ConfigurationError(
                "native MegaMoE needs its measured routed module table; no generic kernel substitution"
            )
        self.sequence_parallel = self.parallel.tensor_parallel_size > 1
        self.fused_shared = bool(
            config.shared_expert_fusion
            and self.model.moe.num_shared_experts
            and self.model.operator_dtype("shared_expert") == "fp8_block"
        )
        self.lookup = OperatorLookup(
            backend.package,
            LookupPolicy(extrapolate="analytical"),
            analytical_scaler=lambda table, key: float(key["num_tokens"]),
        )
        self.base_key = {
            "dtype": self.model.operator_dtype("routed_expert"),
            "kernel_source": "deepgemm_megamoe",
            "kernel_dtype": "fp8_fp4",
            "distribution": "power_law_1.01",
            "pre_dispatch": "sglang_jit",
            "source_policy": "random",
            "hidden_size": self.model.hidden_size,
            "inter_size": self.model.moe_intermediate_size,
            "top_k": self.model.moe.num_experts_per_tok,
            "num_experts": self.model.moe.num_experts,
            "num_fused_shared_experts": 0,
            "tp_size": 1,
            "ep_size": backend.moe_ep,
        }
        selection = binding.get("key", {})
        selectable = {
            "kernel_source",
            "kernel_dtype",
            "distribution",
            "pre_dispatch",
            "source_policy",
        }
        if set(selection) - selectable:
            raise ConfigurationError(
                "MegaMoE bindings select kernel scope; geometry comes from the model"
            )
        self.base_key.update(
            {name: str(value).strip().lower() for name, value in selection.items()}
        )
        rows = [
            r
            for r in backend.package.tables[self.table]
            if all(r[k] == v for k, v in self.base_key.items())
        ]
        if not rows:
            raise ConfigurationError(
                "no native MegaMoE measurements for this model/EP/quantization scope"
            )
        if any(
            r.get("includes_gate_topk") is not False
            or r.get("includes_buffer_init") is not False
            or r.get("used_cuda_graph") is not True
            or r.get("includes_routed_scale") is not True
            for r in rows
        ):
            raise ConfigurationError("native MegaMoE table timing scope not verified")
        self.regimes = {}
        for phase in ("generation", "context"):
            values = sorted(r["num_tokens"] for r in rows if r["phase"] == phase)
            if values:
                self.regimes[phase] = (values[0], values[-1])
        self.cache = BoundedCache(32768)
        self.step = None
        self.topology_checked = False
        collective = getattr(backend.communicator, "collective_model", None)
        links = getattr(collective, "links", None)
        if not isinstance(links, LinkCatalog):
            links = _default_links(default_data_root())
        link = links.get(backend.device.scale_up_link)
        self.work_model = MegaMoEWorkModel(
            self.model,
            backend.device,
            backend.moe_ep,
            nvlink_bytes_per_s=link.bandwidth_per_direction_bytes_per_s,
        )
        from TokenSim.latency.moe_efficiency import ExpertEfficiencyTransfer

        self.efficiency_transfer = ExpertEfficiencyTransfer(self.work_model, rows)
        self.priced = BoundedCache(65536)
        self.reset_stats()

    def reset_stats(self):
        self.counts = Counter()
        self.routed_seconds = self.shared_seconds = 0.0
        self.max_rank_rows = 0
        self.composition = Counter()

    def set_step(self, step):
        self.step = step

    def _query(self, tokens):
        tokens = max(1, tokens)
        cached = self.cache.get(tokens)
        if cached is not None:
            return cached
        generation = self.regimes.get("generation")
        context = self.regimes.get("context")
        if generation and tokens <= generation[1]:
            phase = "generation"
        elif context and tokens >= context[0]:
            phase = "context"
        elif generation and context:
            # Explicit interpolation across collector buffer regimes. Both
            # anchors price the same underlying kernel, but not identical
            # capacity settings; do not mislabel this as same-config data.
            lo, hi = generation[1], context[0]
            a = self.lookup.lookup(
                self.table,
                {**self.base_key, "phase": "generation", "num_tokens": lo},
            )
            b = self.lookup.lookup(
                self.table, {**self.base_key, "phase": "context", "num_tokens": hi}
            )
            fraction = (tokens - lo) / (hi - lo)
            result = ModuleEstimate(
                a.latency_us + fraction * (b.latency_us - a.latency_us),
                "interpolated",
                a.source_id,
                "cross_buffer_regime",
            )
            self.cache[tokens] = result
            return result
        else:
            phase = "generation" if generation else "context"
        query = self.lookup.lookup(
            self.table, {**self.base_key, "phase": phase, "num_tokens": tokens}
        )
        result = ModuleEstimate(
            query.latency_us, query.match_type, query.source_id, phase
        )
        self.cache[tokens] = result
        return result

    def group_rows(self, local_tokens):
        tp, dp = self.parallel.tensor_parallel_size, self.backend.rank_info.dp_rank
        if self.step is None:
            if self.parallel.data_parallel_size > 1:
                raise ConfigurationError(
                    "native MegaMoE needs the synchronized DP batch snapshot"
                )
            step = ForwardPass({dp: RankShape(local_tokens, None, True)})
        else:
            step = self.step
        # FULL_DECODE_ONLY: a context rank forces eager; a staged q=1 decode
        # cannot use the reference MTP q=4 captures. Keep actual query widths.
        padded = uniform_decode_rows(step, self.expected_query_width, self.max_num_seqs)
        graph = padded is not None
        group = self.parallel.expert_parallel_group(dp, self.backend.rank_info.tp_rank)
        rows, valid_rows = {}, {}
        for other_dp, shape in step.ranks.items():
            count = padded if graph else shape.tokens
            for other_tp in range(tp):
                if self.parallel.expert_parallel_group(other_dp, other_tp) != group:
                    continue
                # vLLM pads sequence-parallel tensors to TP divisibility.
                value = math.ceil(count / tp) if self.sequence_parallel else count
                ep_rank = self.parallel.expert_parallel_rank(other_dp, other_tp)
                rows[ep_rank] = value
                valid_rows[ep_rank] = (
                    sp_valid_rows(shape.routed_tokens, count, tp, other_tp)
                    if self.sequence_parallel
                    else shape.routed_tokens
                )
        if len(rows) != self.backend.moe_ep:
            raise ConfigurationError(
                "DP snapshot does not cover the entire MegaMoE EP group"
            )
        rank = self.parallel.expert_parallel_rank(dp, self.backend.rank_info.tp_rank)
        return GroupRows(rows, valid_rows, rank, graph, step)

    def _price(self, rows, routed_rows):
        anchor_rows = max(1, max(rows.values()))
        estimate = self._query(anchor_rows)
        uniform = {rank: anchor_rows for rank in rows}
        source_work = self.work_model.estimate(uniform, uniform)
        target_work = self.work_model.estimate(rows, routed_rows)
        anchor = estimate.latency_us
        exact_uniform = rows == uniform and routed_rows == uniform
        transfer = self.efficiency_transfer.price(
            anchor,
            source_work,
            target_work,
            anchor_rows=anchor_rows,
            phase=estimate.regime,
        )
        routed = anchor if exact_uniform else transfer.latency_us
        return (
            routed,
            [estimate],
            anchor,
            source_work,
            target_work,
            exact_uniform,
            transfer,
        )

    def shared_fused_delta(self, rows):
        # Dedicated native shared MMA uses FP8 weights and packed 1x32
        # scales. Its intermediate is FP8; it aliases the input dispatch
        # buffer. Estimate incremental work with no standalone launches.
        if not self.model.moe.num_shared_experts:
            return 0.0
        h = self.model.hidden_size
        i = self.model.moe_intermediate_size * self.model.moe.num_shared_experts
        device = self.backend.device
        parameters = device.analytical
        effective_rows = max(rows, float(parameters.get("gemm_small_m_knee", 64)))
        compute = (
            6
            * effective_rows
            * h
            * i
            / (
                device.peak_compute_for("fp8_block").value
                * float(parameters.get("gemm_mfu", 0.7))
            )
        )
        weights = 3 * h * i * (1 + 1 / 32)
        # Read shared X, write/read intermediate+scales, and add BF16 Y.
        activations = rows * (h + 2 * i + 2 * math.ceil(i / 128) * 4 + 4 * h)
        memory = (weights + activations) / (
            device.memory_bandwidth_bytes_per_s.value
            * float(parameters.get("memory_efficiency", 0.85))
        )
        activation = 5 * rows * i / device.peak_compute_for("fp32").value
        return (max(compute, memory) + activation) * 1e6

    def estimate(self, local_tokens, *, layers=None):
        if not self.topology_checked and self.backend.communicator is not None:
            description = self.backend.communicator.describe()
            if (
                description["ep_group_link"] not in ("local", "node")
                or description["ep_group_size"] != self.backend.moe_ep
            ):
                raise ConfigurationError(
                    "borrowed MegaMoE curves require a complete single-node EP group"
                )
            self.topology_checked = True
        group = self.group_rows(local_tokens)
        rows, own_rows, graph, step = (
            group.inputs,
            group.inputs[group.own_rank],
            group.graph,
            group.step,
        )
        identity = (
            id(self.backend.package),
            tuple(sorted(rows.items())),
            tuple(sorted(group.routed.items())),
        )
        priced = step.cache.get(identity)
        if priced is None:
            # Pricing is a pure function of the group rows; reuse it across
            # steps as well as across the ranks of one step.
            priced = self.priced.get(identity)
            if priced is None:
                priced = self._price(rows, group.routed)
                self.priced[identity] = priced
            step.cache[identity] = priced
        routed, estimates, anchor, source_work, target_work, exact_uniform, transfer = (
            priced
        )
        for estimate in estimates:
            self.backend.stats.record(self.table, estimate.match)
            self.counts[f"query_regime:{estimate.regime}"] += 1
        if self.fused_shared:
            shared = self.shared_fused_delta(own_rows)
            self.counts["shared:fused_incremental_estimate"] += 1
        elif self.model.moe.num_shared_experts:
            h, i = (
                self.model.hidden_size,
                self.model.moe_intermediate_size * self.model.moe.num_shared_experts,
            )
            dtype = self.model.operator_dtype("shared_expert")
            shared = (
                self.backend.gemm(own_rows, 2 * i, h, dtype=dtype)
                + self.backend.elementwise("swiglu", own_rows, i)
                + self.backend.gemm(own_rows, h, i, dtype=dtype)
            )
            self.counts["shared:serial_measured_gemm_plus_activation"] += 1
        else:
            shared = 0.0
        layers = self.backend.moe_layers_local if layers is None else layers
        if layers < 1:
            raise ConfigurationError("MegaMoE pass needs at least one layer")
        self.routed_seconds += routed * layers * 1e-6
        self.shared_seconds += shared * layers * 1e-6
        self.counts["rank_estimates"] += 1
        self.counts["graph_pass" if graph else "eager_pass"] += 1
        self.counts["full_input_rows"] += own_rows
        self.counts["routed_valid_rows"] += group.routed[group.own_rank]
        self.counts["padding_rows"] += own_rows - group.routed[group.own_rank]
        self.counts["routed_valid_layer_rows"] += group.routed[group.own_rank] * layers
        self.counts["full_input_layer_rows"] += own_rows * layers
        self.counts["zero_routed_rank_passes"] += int(group.routed[group.own_rank] == 0)
        self.counts[
            "uniform_measured_passes" if exact_uniform else "composed_passes"
        ] += 1
        self.composition["uniform_anchor_seconds"] += anchor * layers * 1e-6
        self.composition["composed_routed_seconds"] += routed * layers * 1e-6
        self.composition["anchor_difference_seconds"] += (
            (anchor - routed) * layers * 1e-6
        )
        self.composition["source_work_us"] += source_work.latency_us
        self.composition["target_work_us"] += target_work.latency_us
        self.composition["input_work_us"] += target_work.input_us
        self.composition["expert_work_us"] += target_work.expert_us
        self.composition["dispatch_work_us"] += target_work.dispatch_us
        self.composition["combine_work_us"] += target_work.combine_us
        self.composition["transferred_common_seconds"] += (
            transfer.common_us * layers * 1e-6
        )
        self.composition["transferred_expert_seconds"] += (
            transfer.expert_us * layers * 1e-6
        )
        self.counts[f"efficiency:{transfer.match}"] += 1
        self.composition["source_mma_layer_rows"] += (
            source_work.expert_mma_rows * layers
        )
        self.composition["target_mma_layer_rows"] += (
            target_work.expert_mma_rows * layers
        )
        self.composition["source_expert_flops"] += source_work.expert_flops * layers
        self.composition["target_expert_flops"] += target_work.expert_flops * layers
        self.composition["source_issued_weight_bytes"] += (
            source_work.issued_weight_bytes * layers
        )
        self.composition["target_issued_weight_bytes"] += (
            target_work.issued_weight_bytes * layers
        )
        self.max_rank_rows = max(self.max_rank_rows, max(rows.values()))
        return (routed + shared) * layers, routed, shared

    def stats_dict(self):
        return {
            "native_megamoe_routed_seconds": self.routed_seconds,
            "native_megamoe_shared_seconds": self.shared_seconds,
            "native_megamoe_counts": dict(self.counts),
            "native_megamoe_composition": dict(self.composition),
            "native_megamoe_max_rank_rows": self.max_rank_rows,
        }

    def describe(self):
        return {
            "backend": self.config.backend,
            "table": self.table,
            "query_scope": dict(self.base_key),
            "sources": self.backend.package.describe_sources(self.table, self.base_key),
            "shared_mode": (
                "fused_incremental_estimate" if self.fused_shared else "serial"
            ),
            "shared_dtype": self.model.operator_dtype("shared_expert"),
            "shared_estimate_grade": (
                "D" if self.fused_shared else "mixed measured/analytical"
            ),
            "sequence_parallel": self.sequence_parallel,
            "num_tokens_scope": "rank-local after TP/SP and eligible graph padding",
            "unequal_ranks": "full-input uniform measured anchor normalized by physical expert, dispatch/scan/combine work; no mean-row substitution",
            "valid_rows": "padding top-k=-1; contiguous SP masks; idle/dummy and MTP valid requests separate from full captured rows",
            "work_composition": self.work_model.describe(),
            "expert_phase_efficiency": self.efficiency_transfer.describe(),
            "collector_regime": "generation for small tokens, context for large, explicit interpolation in the gap",
            "routed_scope": "borrowed routed pre-dispatch/dispatch/compute/combine; gate/top-k excluded",
            "buffer_capacity": "collector capacities retained; serving capacity differs",
            "historical_shared_fusion": "unverified; current pinned source supports native FP8 shared fusion",
            "tp_communication": "existing TP approximation; full native attention/SP communication validation pending",
        }
