"""Compose transformer operators, communication and per-rank routed work."""

from __future__ import annotations

import math
from typing import Mapping
from TokenSim.latency.base import query_tokens
from TokenSim.llm.llm_request import Request


class StepComposition:
    """Compose transformer operators, communication and per-rank routed work."""

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
            if model.qk_norm:
                attn_elem_us += self.elementwise(
                    "rmsnorm", T * (self.q_dim_local // model.head_dim), model.head_dim
                ) + self.elementwise(
                    "rmsnorm", T * (self.kv_dim_local // model.head_dim), model.head_dim
                )
            if model.architecture == "gemma4_text":
                attn_elem_us += self.elementwise(
                    "rmsnorm", T * self.kv_heads_local, model.head_dim
                )
                attn_elem_us += self.elementwise("rmsnorm", T, h)
        dense_ffn_gemm_us = 0.0
        dense_ffn_elem_us = 0.0
        if self.dense_layers_local:
            dense_ffn_gemm_us = self.gemm(
                T,
                model.ffn_projection_count * self.inter_local,
                h,
                dtype=model.operator_dtype("dense_ffn"),
            ) + self.gemm(
                T, h, self.inter_local, dtype=model.operator_dtype("dense_ffn")
            )
            dense_ffn_elem_us = self.elementwise("rmsnorm", T, h) + self.elementwise(
                "residual_add", T, h
            )
            if model.gated:
                dense_ffn_elem_us += self.elementwise(
                    model.activation, T, self.inter_local
                )
            if model.architecture == "gemma4_text":
                dense_ffn_elem_us += self.elementwise("rmsnorm", T, h)

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
        if self.is_last_stage and requests:
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

        overhead_us = (
            self.device.step_overhead_us.value if self.include_step_overhead else 0.0
        )

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
        if self.parallel_config.pipeline_parallel_size > 1 and self.include_pp_transfer:
            latency += self.communicator.estimate_pp_stage_transfer(hidden_bytes)
        return latency

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
