"""Explicit DSV4 MTP passes; measured primitives and named analytical gaps."""

from collections import Counter
import math

from TokenSim.errors import ConfigurationError
from TokenSim.latency.base import query_tokens
from TokenSim.latency.deepseek_v4_attention_sol import causal_limited_pairs
from TokenSim.latency.deepseek_v4_gemm import attention_projections
from TokenSim.latency.deepseek_v4_mhc import DeepSeekV4MHC
from TokenSim.latency.deepseek_v4_moe import MegaMoEEstimator, MoEPass, RankShape


class DeepSeekV4MTPEstimator:
    def __init__(self, backend, *, max_num_seqs):
        self.backend = backend
        self.model, self.spec = backend.model, backend.model.deepseek_v4
        if self.spec is None or backend.parallel_config.pipeline_parallel_size != 1:
            raise ConfigurationError("explicit DSV4 MTP currently needs DSV4 PP1")
        self.k = backend.mtp_query_width - 1
        self.max_num_seqs = max_num_seqs
        self.sp = (
            backend.ep_enabled and backend.tp > 1
            and (backend.native_moe is not None or backend.parallel_config.data_parallel_size > 1)
        )
        self.projections = attention_projections(self.model, backend.tp)
        self.mhc = DeepSeekV4MHC(backend) if backend.native_mhc else None
        self.initial_moe = self.decode_moe = None
        if backend.native_moe:
            self.initial_moe = MegaMoEEstimator(
                backend, backend.moe_execution, expected_query_width=backend.mtp_query_width,
                max_num_seqs=max_num_seqs,
            )
            self.decode_moe = MegaMoEEstimator(
                backend, backend.moe_execution, expected_query_width=1,
                max_num_seqs=max_num_seqs,
            )
        self.step = None
        self.reset_stats()

    def reset_stats(self):
        self.counts, self.component_seconds = Counter(), Counter()
        if self.mhc:
            self.mhc.reset_stats()
        for moe in (self.initial_moe, self.decode_moe):
            if moe:
                moe.reset_stats()

    def set_step(self, step):
        self.step = step

    def _request_count(self, shape, *, own_count=None):
        if shape.requests is not None:
            return shape.requests
        if not shape.context_build and shape.query_width:
            return math.ceil(shape.tokens / shape.query_width)
        if own_count is not None:
            return own_count
        raise ConfigurationError("MTP needs request counts for every mixed DP rank")

    def pass_shapes(self, requests):
        actual_rows = sum(query_tokens(r) for r in requests)
        dp = self.backend.rank_info.dp_rank
        if self.step is None:
            if self.backend.parallel_config.data_parallel_size > 1:
                raise ConfigurationError("MTP needs the synchronized DP request snapshot")
            widths = {query_tokens(r) for r in requests}
            context = any(r.is_context_build for r in requests)
            step = MoEPass({dp: RankShape(actual_rows, next(iter(widths)) if len(widths) == 1 and not context else None,
                                        context, len(requests))})
        else:
            step = self.step
        later = MoEPass({rank: RankShape(self._request_count(shape, own_count=len(requests) if rank == dp else None),
                                       1, False, self._request_count(shape, own_count=len(requests) if rank == dp else None))
                         for rank, shape in step.ranks.items()})
        initial_graph = all(not s.context_build and s.query_width == self.k + 1 for s in step.ranks.values())
        initial_upper = max(s.tokens for s in step.ranks.values())
        initial_graph = initial_graph and initial_upper <= self.max_num_seqs * (self.k + 1)
        initial_rows = math.ceil(initial_upper / (self.k + 1)) * (self.k + 1) if initial_graph else actual_rows
        initial_heads = initial_rows // (self.k + 1) if initial_graph else len(requests)
        later_rows = max(s.tokens for s in later.ranks.values())
        # Later draft decodes have their own q=1 FULL_DECODE_ONLY capture.
        later_graph = later_rows <= self.max_num_seqs
        own_later_rows = later_rows if later_graph else len(requests)
        return [(initial_rows, initial_heads, step, initial_graph)] + [
            (own_later_rows, own_later_rows if later_graph else len(requests), later, later_graph)
            for _ in range(self.k - 1)
        ]

    def _stream_us(self, name, bytes_, *, flops=0, dtype="fp32", launch=True):
        d = self.backend.device
        memory = bytes_ / (d.memory_bandwidth_bytes_per_s.value * d.analytical["memory_efficiency"])
        compute = flops / d.peak_compute_for(dtype).value
        self.backend.stats.record(name, "analytical")
        return max(memory, compute) * 1e6 + (d.analytical["kernel_launch_us"] if launch else 0)

    def _swa(self, requests, pass_index):
        pairs = 0
        for req in requests:
            q = query_tokens(req) if pass_index == 0 else 1
            prefix = req.num_computed_tokens if pass_index == 0 else req.num_computed_tokens + query_tokens(req) + pass_index - 1
            pairs += causal_limited_pairs(1, q, prefix, self.model.sliding_window)
        d = self.backend.device
        flops = 4 * self.backend.heads_local * self.model.head_dim * pairs
        math_us = flops / (d.peak_compute_for(self.model.activation_dtype).value * d.analytical["attention_mfu"]) * 1e6
        # SWA has one plain FP8 latent vector, not separate K and V arrays.
        memory_us = pairs * self.model.head_dim / (d.memory_bandwidth_bytes_per_s.value * d.analytical["memory_efficiency"]) * 1e6
        self.backend.stats.record("mtp_swa", "analytical")
        return max(math_us, memory_us) + d.analytical["kernel_launch_us"]

    def _mhc(self, rows):
        if self.mhc:
            return self.mhc.estimate(rows)
        h, hc = self.model.hidden_size, self.spec.hc_mult
        # Two pre/post sites; named conservative stream/gating estimate when
        # the consumer deliberately selects generic mHC or has no module data.
        return self._stream_us("mtp_mhc", 2 * rows * h * (6 * hc + 4),
                               flops=4 * rows * (hc + 3) * hc * hc * h,
                               launch=False) + 4 * self.backend.device.analytical["kernel_launch_us"]

    def _moe(self, rows, step, initial):
        b = self.backend
        estimator = self.initial_moe if initial else self.decode_moe
        if estimator:
            estimator.set_step(step)
            return estimator.estimate(rows, layers=1)[0]
        # Explicit generic kernel path, preserving that backend's group-token
        # convention. It is not substituted for a requested native MegaMoE.
        routed = b._query_moe({"dtype": b.moe_dtype, "hidden_size": self.model.hidden_size,
                              "inter_size": self.model.moe_intermediate_size,
                              "top_k": self.model.moe.num_experts_per_tok,
                              "num_experts": self.model.moe.num_experts, "tp_size": b.moe_tp,
                              "ep_size": b.moe_ep, "num_tokens": max(1, int(rows * b.moe_token_multiplier))}).latency_us
        local = math.ceil(rows / b.tp) if self.sp else rows
        i = self.model.moe_intermediate_size * self.model.moe.num_shared_experts
        shared = (b._gemm(local, 2 * i, self.model.hidden_size, dtype=self.model.operator_dtype("shared_expert"))
                  + b._elementwise("swiglu", local, i)
                  + b._gemm(local, self.model.hidden_size, i, dtype=self.model.operator_dtype("shared_expert"))) if i else 0
        return routed + shared

    def _head(self, rows):
        h, hc = self.model.hidden_size, self.spec.hc_mult
        # hc_head: FP32 gate projection/RMS/reweight, BF16 residual input and
        # dense output. There is no matching full hc_head table in this package.
        weight_bytes = 4 * (hc * hc * h + hc + 1)
        head = self._stream_us("mtp_hc_head", weight_bytes + rows * h * (2 * hc + 2),
                               flops=rows * h * (2 * hc * hc + 8 * hc))
        return head + self.backend._elementwise("rmsnorm", rows, h) + self.backend._gemm(
            rows, self.backend.vocab_local, h, dtype=self.model.operator_dtype("lm_head"))

    def estimate(self, requests):
        if not requests:
            return Counter()
        b, h, hc = self.backend, self.model.hidden_size, self.spec.hc_mult
        totals = Counter()
        for index, (rows, head_rows, step, graph) in enumerate(self.pass_shapes(requests)):
            local = math.ceil(rows / b.tp) if self.sp else rows
            values = Counter()
            # Fused e/h RMS norms run before sequence parallel sharding.
            values["elementwise"] += b._elementwise("rmsnorm", rows * (hc + 1), h)
            values["elementwise"] += self._stream_us("mtp_embedding", rows * h * 4)
            values["gemm"] += b._gemm(local, h, h, dtype=self.model.operator_dtype("attention_gemm"))
            values["gemm"] += b._gemm(local * hc, h, h, dtype=self.model.operator_dtype("attention_gemm"))
            values["gemm"] += sum(b._query(p.table, p.key(rows)).latency_us for p in self.projections)
            values["gemm"] += b._gemm(local, self.model.moe.num_experts, h, dtype=self.model.operator_dtype("router"))
            values["attention"] += self._swa(requests, index)
            values["elementwise"] += self._mhc(local) + 2 * b._elementwise("rmsnorm", local, h)
            values["elementwise"] += b._elementwise("rope", rows, b.q_dim_local + self.model.head_dim)
            values["elementwise"] += self._stream_us("mtp_input_add", local * h * 6 * hc)
            values["moe"] += self._moe(rows, step, index == 0)
            values["lm_head"] += self._head(head_rows)
            if self.sp and b.communicator:
                # attention AG input / RS output and MTP's residual AG after
                # final post. The fused MoE owns its EP dispatch/combine.
                values["comm"] += 1e6 * (
                    b.communicator.estimate_tp_collective(rows * h * 2, operation="all_gather")
                    + b.communicator.estimate_tp_collective(rows * h * 2, operation="reduce_scatter")
                    + b.communicator.estimate_tp_collective(rows * hc * h * 2, operation="all_gather")
                )
            elif b.tp > 1 and b.communicator:
                values["comm"] += b.communicator.estimate_tp_collective(rows * h * 2) * 1e6
            if b.ep_enabled and not self.initial_moe and b.communicator:
                values["comm"] += b.communicator.estimate_ep_all2all(
                    rows * self.model.moe.num_experts_per_tok * h * 2, count=1,
                    num_tokens=rows, hidden_size=h, top_k=self.model.moe.num_experts_per_tok,
                    num_experts=self.model.moe.num_experts, activation_bytes=2) * 1e6
            totals.update(values)
            self.counts["passes"] += 1
            self.counts["initial_passes" if index == 0 else "decode_passes"] += 1
            self.counts["forward_rows"] += rows
            self.counts["e_proj_rows"] += local
            self.counts["h_proj_rows"] += local * hc
            self.counts["logits_rows"] += head_rows
            self.counts["graph_passes" if graph else "eager_passes"] += 1
            for key, value in values.items():
                self.component_seconds[key] += value * 1e-6
        self.counts["target_steps"] += 1
        return totals

    def stats_dict(self):
        return {"native_dsv4_mtp_counts": dict(self.counts),
                "native_dsv4_mtp_component_seconds": dict(self.component_seconds)}

    def describe(self):
        return {"K": self.k, "passes": "full target token batch, then K-1 single-query batches",
                "input_projection": "e_proj local rows; h_proj hc_mult * local rows",
                "attention": "SWA only; measured low-rank GEMMs + analytical plain-FP8 single-latent core",
                "mhc": "measured pre/post num_sites=2" if self.mhc else "explicit generic analytical pre/post",
                "moe": "one native routed/shared layer per pass" if self.initial_moe else "generic routed/shared primitives + EP communication",
                "head": "one row/request (capture padding where FULL); analytic hc_head + norm + BF16 shared lm_head",
                "analytical_gaps_grade": "D: SWA core, embedding/input-add and hc_head; no paired full draft module table"}
