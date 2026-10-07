"""Native DSV4 MegaMoE geometry, measured routed scope and shared MLP cost.

The collector's phase is a buffer/token-grid regime, not a kernel argument.
The common EP pass is bounded by the slowest measured rank-local estimate.
This is a conservative unequal-rank approximation to the uniform benchmark.
"""

from collections import Counter
from dataclasses import dataclass, field
import math

from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.lookup import LookupPolicy, OperatorLookup


@dataclass(frozen=True)
class RankShape:
    tokens: int
    query_width: int | None
    context_build: bool
    requests: int | None = None


@dataclass
class MoEPass:
    # A new object for each synchronized iteration; also memoizes common
    # group pricing so multiple leaders do not repeat the same table walk.
    ranks: dict[int, RankShape]
    cache: dict = field(default_factory=dict)


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
        self.expected_query_width, self.max_num_seqs = (
            expected_query_width,
            max_num_seqs,
        )
        if not self.model.deepseek_v4 or not backend.ep_enabled:
            raise ConfigurationError("native MegaMoE requires DSV4 and expert parallel")
        if self.parallel.pipeline_parallel_size != 1 or backend.moe_ep not in (4, 8):
            raise ConfigurationError(
                "native MegaMoE currently supports the PP1, EP4/EP8 reference configurations"
            )
        if (
            backend.fallback == "analytical_only"
            or backend.package is None
            or not backend.package.tables.get("dsv4_megamoe")
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
        rows = [
            r
            for r in backend.package.tables["dsv4_megamoe"]
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
        self.cache = {}
        self.step = None
        self.topology_checked = False
        self.reset_stats()

    def reset_stats(self):
        self.counts = Counter()
        self.routed_seconds = self.shared_seconds = 0.0
        self.max_rank_rows = 0

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
                "dsv4_megamoe",
                {**self.base_key, "phase": "generation", "num_tokens": lo},
            )
            b = self.lookup.lookup(
                "dsv4_megamoe", {**self.base_key, "phase": "context", "num_tokens": hi}
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
            "dsv4_megamoe", {**self.base_key, "phase": phase, "num_tokens": tokens}
        )
        result = ModuleEstimate(
            query.latency_us, query.match_type, query.source_id, phase
        )
        if len(self.cache) > 32768:
            self.cache.clear()
        self.cache[tokens] = result
        return result

    def rows_for_group(self, local_tokens):
        tp, dp = self.parallel.tensor_parallel_size, self.backend.rank_info.dp_rank
        if self.step is None:
            if self.parallel.data_parallel_size > 1:
                raise ConfigurationError(
                    "native MegaMoE needs the synchronized DP batch snapshot"
                )
            step = MoEPass({dp: RankShape(local_tokens, None, True)})
        else:
            step = self.step
        # FULL_DECODE_ONLY: a context rank forces eager; a staged q=1 decode
        # cannot use the reference MTP q=4 captures. Keep actual query widths.
        graph = all(
            not s.context_build and s.query_width == self.expected_query_width
            for s in step.ranks.values()
        )
        upper = max(s.tokens for s in step.ranks.values())
        if upper > self.max_num_seqs * self.expected_query_width:
            graph = False
        padded = (
            self.expected_query_width * math.ceil(upper / self.expected_query_width)
            if graph
            else None
        )
        group = self.parallel.expert_parallel_group(dp, self.backend.rank_info.tp_rank)
        rows = {}
        for other_dp, shape in step.ranks.items():
            count = padded if graph else shape.tokens
            for other_tp in range(tp):
                if self.parallel.expert_parallel_group(other_dp, other_tp) != group:
                    continue
                # vLLM pads sequence-parallel tensors to TP divisibility.
                value = math.ceil(count / tp) if self.sequence_parallel else count
                ep_rank = self.parallel.expert_parallel_rank(other_dp, other_tp)
                rows[ep_rank] = value
        if len(rows) != self.backend.moe_ep:
            raise ConfigurationError(
                "DP snapshot does not cover the entire MegaMoE EP group"
            )
        rank = self.parallel.expert_parallel_rank(dp, self.backend.rank_info.tp_rank)
        return rows, rows[rank], graph, step

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
        rows, own_rows, graph, step = self.rows_for_group(local_tokens)
        identity = (
            id(self.backend.package),
            tuple(sorted(rows.items())),
            self.config.latency_scale,
        )
        if identity not in step.cache:
            estimates = [self._query(n) for n in sorted(set(rows.values()))]
            step.cache[identity] = (
                max(e.latency_us for e in estimates) * self.config.latency_scale,
                estimates,
            )
        routed, estimates = step.cache[identity]
        for estimate in estimates:
            self.backend.stats.record("dsv4_megamoe", estimate.match)
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
                self.backend._gemm(own_rows, 2 * i, h, dtype=dtype)
                + self.backend._elementwise("swiglu", own_rows, i)
                + self.backend._gemm(own_rows, h, i, dtype=dtype)
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
        self.max_rank_rows = max(self.max_rank_rows, max(rows.values()))
        return (routed + shared) * layers, routed, shared

    def stats_dict(self):
        return {
            "native_megamoe_routed_seconds": self.routed_seconds,
            "native_megamoe_shared_seconds": self.shared_seconds,
            "native_megamoe_counts": dict(self.counts),
            "native_megamoe_max_rank_rows": self.max_rank_rows,
        }

    def describe(self):
        return {
            "backend": self.config.backend,
            "latency_scale": self.config.latency_scale,
            "shared_mode": (
                "fused_incremental_estimate" if self.fused_shared else "serial"
            ),
            "shared_dtype": self.model.operator_dtype("shared_expert"),
            "shared_estimate_grade": (
                "D" if self.fused_shared else "mixed measured/analytical"
            ),
            "sequence_parallel": self.sequence_parallel,
            "num_tokens_scope": "rank-local after TP/SP and eligible graph padding",
            "unequal_ranks": "common maximum measured estimate; conservative uniform-rank surrogate",
            "collector_regime": "generation for small tokens, context for large, explicit interpolation in the gap",
            "routed_scope": "borrowed routed pre-dispatch/dispatch/compute/combine; gate/top-k excluded",
            "buffer_capacity": "collector capacities retained; serving capacity differs",
            "historical_shared_fusion": "unverified; current pinned source supports native FP8 shared fusion",
            "tp_communication": "existing TP approximation; full native attention/SP communication validation pending",
        }
