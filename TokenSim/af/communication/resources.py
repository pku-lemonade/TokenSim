"""Physical communication resources and atomic FIFO reservations."""

from __future__ import annotations

from dataclasses import dataclass
import math
import heapq

import simpy

from TokenSim.af.caching.estimates import snapshot
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.config.cluster_config import ClusterConfig, WorkerConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import (
    CommunicationDomainConfig,
    DomainResource,
    physical_domain_config,
    transfer_resources,
)
from TokenSim.hardware.placement import validate_domain_placement


@dataclass(eq=False)
class DomainRequest:
    keys: tuple[str, ...]
    priority: tuple
    event: simpy.Event
    acquired: bool = False
    cancelled: bool = False


class DomainScheduler:
    def __init__(self, env, resource_ids):
        self.env = env
        self.resource_ids = frozenset(resource_ids)
        self.owners = {}
        self._waiters = {}
        self._priorities = set()
        self._resource_queues = {}
        self._serial = 0
        self._scheduled = False

    def request(self, keys, identity):
        keys = tuple(sorted(set(keys)))
        if not keys or not set(keys) <= self.resource_ids:
            raise ConfigurationError(f"invalid communication domain request: {keys!r}")
        if any(type(part) not in (int, str) for part in identity):
            raise ConfigurationError(
                "domain request identity must contain stable integer/string components"
            )
        stable_identity = tuple((type(part).__name__, part) for part in identity)
        request = DomainRequest(keys, (self.env.now, stable_identity), self.env.event())
        if request.priority in self._priorities:
            raise ConfigurationError(
                "concurrent domain requests need unique stable logical identities"
            )
        self._waiters[id(request)] = request
        self._priorities.add(request.priority)
        self._serial += 1
        for key in keys:
            heapq.heappush(
                self._resource_queues.setdefault(key, []),
                (request.priority, self._serial, request),
            )
        self._wake()
        return request

    def _wake(self):
        if not self._scheduled:
            self._scheduled = True
            # A later priority than normal SimPy events gathers every release
            # and readiness event at this timestamp before tie arbitration.
            event = simpy.Event(self.env)
            event._ok, event._value = True, None
            event.callbacks.append(lambda _: self._arbitrate())
            self.env.schedule(event, priority=2, delay=0)

    def _arbitrate(self):
        self._scheduled = False
        candidates = {
            id(request): request
            for key in self._resource_queues
            if (request := self._front(key)) is not None
        }
        for request in sorted(candidates.values(), key=lambda item: item.priority):
            if any(
                key in self.owners or self._front(key) is not request
                for key in request.keys
            ):
                continue
            request.acquired = True
            self._waiters.pop(id(request))
            self._priorities.remove(request.priority)
            for key in request.keys:
                self.owners[key] = request
                heapq.heappop(self._resource_queues[key])
            request.event.succeed()

    def _front(self, key):
        queue = self._resource_queues[key]
        while queue and queue[0][2].cancelled:
            heapq.heappop(queue)
        return queue[0][2] if queue else None

    @property
    def waiters(self):
        return sorted(self._waiters.values(), key=lambda item: item.priority)

    def release(self, request):
        if request.acquired:
            for key in request.keys:
                if self.owners.get(key) is request:
                    del self.owners[key]
            request.acquired = False
        else:
            request.cancelled = True
            self._waiters.pop(id(request), None)
            self._priorities.discard(request.priority)
        self._wake()

    def run(self, keys, duration, identity):
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ConfigurationError("domain service must be finite and nonnegative")
        ready = self.env.now
        if duration == 0:
            return ready, ready, ready
        request = self.request(keys, identity)
        try:
            yield request.event
            start = self.env.now
            yield self.env.timeout(duration)
            return ready, start, self.env.now
        finally:
            self.release(request)


class AFResourceRegistry:
    def __init__(
        self,
        env: simpy.Environment,
        cluster_config: ClusterConfig,
        worker_configs: list[WorkerConfig],
        placement=None,
    ) -> None:
        physical = physical_domain_config(cluster_config.af)
        if physical is not None:
            validate_domain_placement(placement, worker_configs, physical)
        self.placement = snapshot(placement)
        self.env = env
        self.cluster_config = cluster_config
        self.worker_configs = {
            index: config for index, config in enumerate(worker_configs)
        }
        self.worker_ids_by_physical_identity = {}
        for worker_id, config in self.worker_configs.items():
            if config.physical_worker_ref is None:
                raise ConfigurationError(
                    f"worker {worker_id} has no physical worker identity"
                )
            identity = config.physical_worker_ref.identity
            if identity in self.worker_ids_by_physical_identity:
                raise ConfigurationError(
                    f"duplicate physical worker identity {identity!r}"
                )
            self.worker_ids_by_physical_identity[identity] = worker_id
        self.compute_slots = {
            worker_id: simpy.Resource(env, capacity=1)
            for worker_id in self.worker_configs
        }
        from TokenSim.af.communication.endpoints import EndpointResources

        self.endpoint_resources = (
            EndpointResources(
                env, cluster_config.networks, self.worker_ids_by_physical_identity
            )
            if cluster_config.af.communication_resource_scope == "endpoint"
            else None
        )
        timing = cluster_config.af.communication_timing
        host_keys = (
            {"shared"}
            if timing.host_resource_scope == "shared"
            else {str(worker_id) for worker_id in self.worker_configs}
        )
        self.host_submission_slots = (
            {key: simpy.Resource(env, capacity=1) for key in host_keys}
            if timing.enabled
            or cluster_config.af.compute_execution_mode == "cuda_graph"
            else {}
        )
        self.group_worker_ids: dict[str, tuple[int, ...]] = {}
        for worker_id, worker in self.worker_configs.items():
            if worker.name is None:
                raise ConfigurationError(
                    f"worker {worker_id} has no normalized group name"
                )
            self.group_worker_ids.setdefault(worker.name, ())
            self.group_worker_ids[worker.name] += (worker_id,)

        self.communication_resources = {
            key: simpy.Resource(env, capacity=config.capacity)
            for key, config in cluster_config.af_communication_resources.items()
        }
        config = cluster_config.af.communication_domains or CommunicationDomainConfig()
        self.node_by_worker = (
            {
                worker_id: self.placement.node_id(worker_id)
                for worker_id in self.worker_configs
            }
            if physical is not None
            else {
                worker_id: (
                    placement.device_index(worker_id)
                    if placement is not None
                    else config.device_indices.get(
                        worker.physical_worker_ref.identity,
                        (
                            worker.device_index
                            if worker.device_index is not None
                            else worker_id
                        ),
                    )
                )
                // config.node_size
                for worker_id, worker in self.worker_configs.items()
            }
        )
        nodes = (
            set(range(config.node_count))
            if config.node_count
            else set(self.node_by_worker.values())
        )
        if not set(self.node_by_worker.values()) <= nodes:
            raise ConfigurationError(
                "AF worker placement exceeds communication domain inventory"
            )
        self.domain_resources = {
            DomainResource(node, kind).identity: simpy.Resource(env, capacity=1)
            for node in sorted(nodes)
            for kind in ("local", "inter_tx", "inter_rx")
        }
        self.domain_scheduler = DomainScheduler(env, self.domain_resources)
        configured_groups = {key[0] for key in self.communication_resources}
        for domain in cluster_config.af.execution_domains:
            for collective in domain.communication_plan:
                if (
                    collective.resource_group is not None
                    and collective.resource_group not in configured_groups
                ):
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} collective "
                        f"{collective.operation!r} references missing fabric alias "
                        f"resource_group={collective.resource_group!r}"
                    )

    def worker(self, worker_id: int) -> WorkerConfig:
        try:
            return self.worker_configs[worker_id]
        except KeyError as exc:
            raise ConfigurationError(f"unknown AF worker id {worker_id}") from exc

    def group_for_worker(self, worker_id: int) -> str:
        worker = self.worker(worker_id)
        if worker.name is None:
            raise ConfigurationError(f"worker {worker_id} has no group")
        return worker.name

    def worker_id_for_physical(self, identity: str) -> int:
        try:
            return self.worker_ids_by_physical_identity[identity]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown physical worker identity {identity!r}"
            ) from exc

    def compute_resources_for(
        self, physical_identities: tuple[str, ...]
    ) -> tuple[tuple[int, simpy.Resource], ...]:
        if len(physical_identities) != len(set(physical_identities)):
            raise ConfigurationError(
                "compute step contains duplicate physical workers; execution-domain "
                "placement must be strictly one-to-one"
            )
        resolved = tuple(
            (
                self.worker_id_for_physical(identity),
                self.compute_slots[self.worker_id_for_physical(identity)],
            )
            for identity in physical_identities
        )
        return tuple(sorted(resolved))

    def communication_resources_for_step(
        self,
        link: AFLinkConfig,
        direction: str,
        resource_group: str | None = None,
        resource_lane: str | None = None,
        physical_endpoints: tuple[str, ...] = (),
        network: str | None = None,
    ) -> tuple[tuple[tuple[str, str], simpy.Resource], ...]:
        if self.endpoint_resources is not None:
            return self.endpoint_resources.for_step(
                network or link.network, direction, physical_endpoints
            )
        if self.cluster_config.af.transfer_topology_mode == "physical":
            workers = tuple(
                self.worker_id_for_physical(identity) for identity in physical_endpoints
            )
            if len(workers) < 2:
                return ()
            keys = set()
            for source, target in zip(workers, workers[1:]):
                keys.update(
                    resource.identity
                    for resource in transfer_resources(
                        self.node_by_worker[source], self.node_by_worker[target]
                    )
                )
            return tuple((key, self.domain_resources[key]) for key in sorted(keys))
        group = resource_group or link.resource_group
        lane = resource_lane
        if lane is None:
            if direction not in {"a_to_f", "f_to_a"}:
                raise ConfigurationError(
                    f"AF communication operation {direction!r} requires an explicit "
                    "resource_lane; collective resources must not be inferred from "
                    "an AF transfer direction"
                )
            lane = "shared" if link.duplex == "half" else direction
        key = (group, lane)
        resource = self.communication_resources.get(key)
        if resource is None:
            raise ConfigurationError(
                f"AF communication resource {key!r} is missing for "
                f"operation={direction!r}; the execution plan must resolve an "
                "existing resource_group/resource_lane"
            )
        return ((key, resource),)

    def resource_capacities(self) -> dict[str, int]:
        capacities = {f"worker:{worker_id}": 1 for worker_id in self.compute_slots}
        capacities.update({f"host:{key}": 1 for key in self.host_submission_slots})
        capacities.update(
            {
                self.format_link_resource(key): resource.capacity
                for key, resource in self.communication_resources.items()
            }
        )
        if self.endpoint_resources is not None:
            capacities.update(self.endpoint_resources.capacities())
        if self.cluster_config.af.transfer_topology_mode == "physical":
            capacities = {
                key: value
                for key, value in capacities.items()
                if not key.startswith("link:")
            }
            capacities.update({key: 1 for key in self.domain_resources})
        return capacities

    def host_resources_for(self, physical_identities):
        scope = self.cluster_config.af.communication_timing.host_resource_scope
        keys = (
            {"shared"}
            if scope == "shared"
            else {
                str(self.worker_id_for_physical(identity))
                for identity in physical_identities
            }
        )
        return tuple(
            (f"host:{key}", self.host_submission_slots[key]) for key in sorted(keys)
        )

    @staticmethod
    def format_link_resource(key: tuple[str, str]) -> str:
        if isinstance(key, str):
            return key
        return f"link:{key[0]}:{key[1]}"
