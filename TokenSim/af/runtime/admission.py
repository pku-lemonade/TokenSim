"""Resident KV accounting for fixed serving cohorts on compiled AF ranks."""

from dataclasses import dataclass

from TokenSim.config.cache_config import stage_layer_range
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError, SimulationStateError


def rank_layer_kv_bytes(plan, binding, context):
    """Return bounded-window, locally owned KV residence per layer on one rank."""
    integer(context, "KV context")
    if binding.kv_partition is None:
        return {}
    domain = plan.domain_runtime(binding.domain)
    layers = tuple(
        stage_layer_range(
            plan.total_layers,
            domain.pipeline_parallel_size,
            binding.pp_rank,
            domain.pipeline_layer_counts,
        )
    )
    per_layer = binding.kv_partition.bytes_per_token // len(layers)
    return {
        layer: (
            binding.kv_partition.for_layer(layer).bytes_per_token
            if binding.kv_partition.layer_partitions
            else per_layer
        )
        * binding.kv_partition.for_layer(layer).resident_tokens(
            context, plan.layer_plan(layer).attention_window
        )
        for layer in layers
    }


def rank_kv_bytes(plan, binding, context):
    return sum(rank_layer_kv_bytes(plan, binding, context).values())


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
        self.reserved_by_rank = {}

    def _rank_bytes(self, domain, instance, context):
        return {
            binding.rank_key: rank_kv_bytes(self.plan, binding, context)
            for binding in self.plan.bindings_for_instance(domain, instance)
        }

    def _bytes(self, domain, instance, context):
        result = {}
        for rank, size in self._rank_bytes(domain, instance, context).items():
            binding = self.plan.binding(*rank)
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
        rank_sizes = self._rank_bytes(domain, instance, maximum_context)
        replica = self.plan.replica_for_instance(domain, instance)
        for rank, blocks in replica.local_gpu_blocks:
            partition = self.plan.binding(*rank).kv_partition
            budget = blocks * partition.bytes_per_block if partition else 0
            required = self.reserved_by_rank.get(rank, 0) + rank_sizes[rank]
            if required > budget:
                raise ConfigurationError(
                    f"resident KV exceeds compiled replica budget on {rank}: "
                    f"required_bytes={required}, available_bytes={budget}"
                )
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
        for rank, size in rank_sizes.items():
            self.reserved_by_rank[rank] = self.reserved_by_rank.get(rank, 0) + size

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
        for rank, size in self._rank_bytes(
            lease.domain, lease.instance, lease.maximum_context
        ).items():
            self.reserved_by_rank[rank] -= size

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
