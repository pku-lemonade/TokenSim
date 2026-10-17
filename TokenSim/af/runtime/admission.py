"""Resident KV accounting for fixed serving cohorts on compiled AF ranks."""

from dataclasses import dataclass

from TokenSim.config.cache_config import stage_layer_range
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError, SimulationStateError


def rank_kv_bytes(plan, binding, context):
    integer(context, "KV context")
    if binding.kv_partition is None:
        return 0
    domain = plan.domain_runtime(binding.domain)
    layers = tuple(
        stage_layer_range(
            plan.total_layers, domain.pipeline_parallel_size, binding.pp_rank
        )
    )
    per_layer = binding.kv_partition.bytes_per_token // len(layers)
    return sum(
        per_layer * min(context, plan.layer_plan(layer).attention_window or context)
        for layer in layers
    )


@dataclass(frozen=True)
class KVLease:
    request_id: int
    domain: str
    instance: str
    maximum_context: int
    current_context: int


class ResidentKV:
    """Validate peak cohort residence before launching model work.

    The cohort executor has no paging or preemption. Reservations are explicit
    byte leases on all required physical KV shards, including bounded windows.
    """

    def __init__(self, plan):
        self.plan = plan
        self.leases = {}
        self.reserved = {state.physical_identity: 0 for state in plan.physical_states}

    def _bytes(self, domain, instance, context):
        result = {}
        for binding in self.plan.bindings_for_instance(domain, instance):
            size = rank_kv_bytes(self.plan, binding, context)
            result[binding.physical_identity] = (
                result.get(binding.physical_identity, 0) + size
            )
        return result

    def admit(self, request_id, domain, instance, *, maximum_context, current_context):
        integer(request_id, "request ID")
        integer(maximum_context, "maximum KV context", minimum=1)
        integer(current_context, "current KV context")
        if request_id in self.leases or current_context > maximum_context:
            raise ConfigurationError(
                "KV lease must have a unique request and bounded context"
            )
        sizes = self._bytes(domain, instance, maximum_context)
        for physical, size in sizes.items():
            state = self.plan.physical_state(physical)
            required = (
                state.model_bytes
                + state.protocol_buffer_bytes
                + self.reserved[physical]
                + size
            )
            if required > state.capacity_bytes:
                raise ConfigurationError(
                    f"resident KV does not fit on {physical}: required={required}, available={state.capacity_bytes}"
                )
        self.leases[request_id] = KVLease(
            request_id, domain, instance, maximum_context, current_context
        )
        for physical, size in sizes.items():
            self.reserved[physical] += size

    def commit(self, request_id, context):
        integer(context, "committed KV context")
        lease = self.leases[request_id]
        if context > lease.maximum_context:
            raise SimulationStateError("KV commit exceeds the admitted reservation")
        from dataclasses import replace

        self.leases[request_id] = replace(lease, current_context=context)

    def release(self, request_id):
        lease = self.leases.pop(request_id)
        for physical, size in self._bytes(
            lease.domain, lease.instance, lease.maximum_context
        ).items():
            self.reserved[physical] -= size

    def summary(self):
        current = {physical: 0 for physical in self.reserved}
        for lease in self.leases.values():
            for physical, size in self._bytes(
                lease.domain, lease.instance, lease.current_context
            ).items():
                current[physical] += size
        return {
            "requests": len(self.leases),
            "reserved_peak_kv_bytes": dict(self.reserved),
            "resident_current_kv_bytes": current,
        }
