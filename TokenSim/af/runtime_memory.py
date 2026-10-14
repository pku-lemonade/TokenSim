"""Runtime memory."""

from __future__ import annotations

from TokenSim.af.runtime_records import (
    AFModelComponent,
    AFPhysicalRuntimeState,
    RankKey,
)
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


def _compile_physical_states(
    *,
    preliminary,
    kv_by_rank,
    worker_by_physical,
    hardware,
    usable_memory_fraction,
    cluster_config,
    moe_config,
    hidden,
    effective_dtypes,
    model,
):
    physical_components: dict[str, dict[str, AFModelComponent]] = {}
    physical_bindings: dict[str, list[RankKey]] = {}
    physical_kv_bytes: dict[str, list[int]] = {}
    for item in preliminary:
        domain = item["domain"]
        mapping = item["mapping"]
        rank_key = (domain.name, mapping.domain_rank)
        physical = mapping.physical_worker.identity
        physical_bindings.setdefault(physical, []).append(rank_key)
        ledger = physical_components.setdefault(physical, {})
        for component in item["components"]:
            ledger.setdefault(component.component_id, component)
        partition = kv_by_rank.get(rank_key)
        if partition is not None:
            physical_kv_bytes.setdefault(physical, []).append(partition.bytes_per_block)

    physical_states: list[AFPhysicalRuntimeState] = []
    for physical, rank_keys in sorted(physical_bindings.items()):
        worker = worker_by_physical[physical]
        try:
            hardware_conf = hardware.device(worker.hardware)
            capacity_bytes = int(
                hardware_conf.usable_memory_bytes * usable_memory_fraction
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"AF runtime hardware capacity metadata missing for "
                f"worker={physical!r}, hardware={worker.hardware!r}"
            ) from exc
        components = tuple(
            physical_components[physical][component_id]
            for component_id in sorted(physical_components[physical])
        )
        model_bytes = sum(component.bytes for component in components)
        buffers = ()
        if cluster_config.af.transfer_topology_mode == "physical":
            # Explicit conservative workspace assumption: one input and one
            # output buffer for the largest activation/logit boundary in a
            # configured AF block. Reused across layers, not silently zero.
            routing_expansion = (
                max(1, moe_config.num_experts_per_tok) if moe_config is not None else 1
            )
            workspace = (
                2
                * cluster_config.af.af_block_size
                * max(hidden * routing_expansion, model.vocab_size)
                * dtype_size_bytes(
                    effective_dtypes.activation_dtype, role="activation_dtype"
                )
            )
            buffers += (("communication_workspace", workspace),)
        if (
            cluster_config.af.execution_protocol == "vllm_afd"
            and cluster_config.af.communication_timing.native_control_enabled
        ):
            from TokenSim.af.control import encode_eager_control

            tokens = cluster_config.af.af_block_size
            hidden_bytes = (
                tokens
                * hidden
                * dtype_size_bytes(
                    effective_dtypes.activation_dtype, role="activation_dtype"
                )
            )
            roles = {
                item["domain"].capability
                for item in preliminary
                if item["mapping"].physical_worker.identity == physical
            }
            # Dedicated logical protocol buffers for one eager iteration, reused
            # across layers. Kernel workspaces/allocator fragmentation are separate.
            if "af_attention" in roles:
                buffers += (
                    ("a_residual", hidden_bytes),
                    ("a_send_hidden", hidden_bytes),
                    ("a_return_hidden", hidden_bytes),
                )
            if "af_ffn" in roles:
                buffers += (
                    ("f_input_hidden", hidden_bytes),
                    ("f_output_hidden", hidden_bytes),
                    ("f_router_logits", tokens * moe_config.num_experts * 2),
                    ("f_topk_ids_weights", tokens * moe_config.num_experts_per_tok * 8),
                )
            buffers += (("control_staging", 8 + len(encode_eager_control(tokens))),)
        required_bytes = model_bytes + sum(size for _, size in buffers)
        if required_bytes >= capacity_bytes:
            domains_and_ranks = ",".join(
                f"{domain}:{rank}" for domain, rank in rank_keys
            )
            component_bytes: dict[str, int] = {}
            for component in components:
                component_bytes[component.category] = (
                    component_bytes.get(component.category, 0) + component.bytes
                )
            raise ConfigurationError(
                f"AF model does not fit before execution: domain/rank={domains_and_ranks}, "
                f"physical_worker={physical}, hardware={worker.hardware}, "
                f"component_bytes={component_bytes}, model_bytes={model_bytes}, "
                f"capacity_bytes={capacity_bytes}, required_bytes={required_bytes}, "
                f"available_bytes={capacity_bytes}"
            )
        physical_states.append(
            AFPhysicalRuntimeState(
                physical_identity=physical,
                hardware=worker.hardware,
                capacity_bytes=capacity_bytes,
                bindings=tuple(sorted(rank_keys)),
                components=components,
                model_bytes=model_bytes,
                kv_bytes_per_block=max(physical_kv_bytes.get(physical, [0])),
                protocol_buffers=buffers,
            )
        )
    return tuple(physical_states)
