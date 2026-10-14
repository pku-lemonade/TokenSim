"""Planner router."""

from __future__ import annotations

from TokenSim.af.planning.nodes import _TransferCopy
from TokenSim.af.providers.queries import AFTensorSlice
from TokenSim.dtypes import dtype_size_bytes


class RouterLogitsMixin:
    def _append_router_logits(
        self,
        *,
        nodes,
        tokens,
        a_leader,
        f_service_bindings,
        router_id,
        layer,
        link,
        prefix,
        layer_id,
        arrivals,
    ):
        dtype = self.runtime_plan.dtypes_for_link(link.name).a_to_f_dtype
        target = f_service_bindings[0]
        logits_copies = tuple(
            _TransferCopy(
                source_binding=a_leader,
                target_binding=target,
                tensor_slice=AFTensorSlice(
                    tensor_item_id=f"request:{request}:token:{token}:router_logits",
                    interval=(0, layer.num_experts),
                    layout="replicated",
                    owner_domain=a_leader.domain,
                    owner_domain_rank=a_leader.domain_rank,
                    owner_dp_rank=a_leader.dp_rank,
                    owner_tp_rank=a_leader.tp_rank,
                    owner_physical_worker=a_leader.physical_identity,
                    canonical_producer=router_id,
                    element_count=layer.num_experts,
                    payload_bytes=layer.num_experts
                    * dtype_size_bytes(dtype, role="a_to_f_dtype"),
                ),
                predecessor_id=router_id,
                consumer_id="expert_logits",
                source_layout="replicated",
                target_layout="replicated",
            )
            for request, token in tokens
        )
        logits_arrivals = self._append_transfer_stage(
            nodes,
            logical_stage_id=f"{prefix}:router_logits",
            operation="af_router_logits",
            direction="a_to_f",
            copies=logits_copies,
            link=link,
            layer_type="moe",
            layer_id=layer_id,
        )
        arrivals = {
            name: tuple(dict.fromkeys((*preds, *logits_arrivals["expert_logits"])))
            for name, preds in arrivals.items()
        }
        return arrivals
