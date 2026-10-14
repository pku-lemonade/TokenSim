"""Operator validation."""

from __future__ import annotations

import math

from TokenSim.af.providers.queries import AFComputeQuery


class ComputeValidationMixin:
    def _validate_local_work(self, query: AFComputeQuery) -> tuple[int, int]:
        context = (
            f"operation={query.operation!r}, stage={query.stage!r}, "
            f"domain={query.source_domain!r}, dp_rank={query.dp_rank}, "
            f"tp_rank={query.tp_rank}, ep_rank={query.ep_rank}"
        )
        if query.local_token_count is None or query.local_token_count < 1:
            raise ValueError(
                f"AF compute query missing positive local_token_count; {context}"
            )
        if query.token_count != query.local_token_count:
            raise ValueError(
                f"AF compute query token_count={query.token_count} does not match "
                f"local_token_count={query.local_token_count}; {context}"
            )
        shape = query.local_shape
        required = ("tokens", "hidden_size", "global_hidden_size", "tp_size")
        missing = sorted(set(required) - set(shape))
        if missing:
            raise ValueError(
                f"AF compute query missing local_shape fields {missing}; {context}"
            )
        values = {name: shape[name] for name in required}
        if any(type(value) is not int or value < 1 for value in values.values()):
            raise ValueError(
                f"AF compute query local_shape must contain positive integers; "
                f"local_shape={dict(shape)!r}; {context}"
            )
        if values["tokens"] != query.local_token_count:
            raise ValueError(
                f"AF compute query local_shape tokens={values['tokens']} does not "
                f"match local_token_count={query.local_token_count}; {context}"
            )
        if values["global_hidden_size"] != self.model.hidden_size:
            raise ValueError(
                f"AF compute query global_hidden_size={values['global_hidden_size']} "
                f"does not match model hidden_size={self.model.hidden_size}; {context}"
            )
        expected_hidden = (
            values["hidden_size"]
            if query.stage in {"encode", "decode"}
            else (
                self.model.hidden_size
                if query.stage
                in {
                    "router",
                    "attention",
                    "dense_ffn",
                    "shared_expert",
                    "routed_expert",
                }
                or (
                    query.stage == "layout"
                    and query.operation
                    in {
                        "post_attention_norm",
                        "embedding_lookup",
                        "final_norm",
                        "lm_head",
                        "token_selection",
                    }
                )
                else math.ceil(self.model.hidden_size / values["tp_size"])
            )
        )
        if values["hidden_size"] != expected_hidden:
            raise ValueError(
                f"AF compute query local hidden_size={values['hidden_size']} does not "
                f"match expected={expected_hidden}; {context}"
            )
        return query.local_token_count, values["tp_size"]

    @staticmethod
    def _validate_expert_workload(
        query: AFComputeQuery, tokens: int, num_experts: int
    ) -> tuple[tuple[int, int], ...]:
        workload = tuple(query.expert_workload)
        if not workload:
            raise ValueError(
                "MoE expert_rank_compute requires non-empty expert_workload; "
                f"domain={query.source_domain!r}, ep_rank={query.ep_rank}, "
                f"local_token_count={tokens}"
            )
        expert_ids = [expert_id for expert_id, _ in workload]
        if len(expert_ids) != len(set(expert_ids)):
            raise ValueError("expert_workload contains duplicate expert ids")
        if any(expert_id < 0 or count < 1 for expert_id, count in workload):
            raise ValueError(
                "expert_workload requires non-negative ids and positive counts"
            )
        if any(expert_id >= num_experts for expert_id in expert_ids):
            raise ValueError(
                f"expert_workload contains expert id outside model range [0, {num_experts})"
            )
        if tuple(query.expert_ids) != tuple(expert_ids):
            raise ValueError(
                f"expert_ids={tuple(query.expert_ids)!r} does not match "
                f"expert_workload ids={tuple(expert_ids)!r}"
            )
        workload_tokens = sum(count for _, count in workload)
        if workload_tokens != tokens:
            raise ValueError(
                f"expert_workload total={workload_tokens} does not match local_token_count={tokens}"
            )
        return workload
