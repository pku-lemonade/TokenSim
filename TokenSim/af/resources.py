from __future__ import annotations

import simpy

from TokenSim.config.cluster_config import AFLinkConfig, ClusterConfig, WorkerConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import CommunicationDomainConfig, DomainResource, transfer_resources
from TokenSim.af.domain_scheduler import DomainScheduler


class AFResourceRegistry:
    def __init__(
        self,
        env: simpy.Environment,
        cluster_config: ClusterConfig,
        worker_configs: list[WorkerConfig],
        placement=None,
    ) -> None:
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
        timing = cluster_config.af.communication_timing
        host_keys = ({"shared"} if timing.host_resource_scope == "shared"
                     else {str(worker_id) for worker_id in self.worker_configs})
        self.host_submission_slots = {
            key: simpy.Resource(env, capacity=1) for key in host_keys
        } if timing.enabled else {}
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
        self.node_by_worker = {
            worker_id: (placement.device_index(worker_id) if placement is not None else
                        config.device_indices.get(worker.physical_worker_ref.identity,
                                                  worker.device_index if worker.device_index is not None else worker_id)) // config.node_size
            for worker_id, worker in self.worker_configs.items()
        }
        nodes = set(range(config.node_count)) if config.node_count else set(self.node_by_worker.values())
        if not set(self.node_by_worker.values()) <= nodes:
            raise ConfigurationError("AF worker placement exceeds communication domain inventory")
        self.domain_resources = {
            DomainResource(node, kind).identity: simpy.Resource(env, capacity=1)
            for node in sorted(nodes) for kind in ("local", "inter_tx", "inter_rx")
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
    ) -> tuple[tuple[tuple[str, str], simpy.Resource], ...]:
        if self.cluster_config.af.transfer_topology_mode == "physical":
            workers = tuple(self.worker_id_for_physical(identity) for identity in physical_endpoints)
            if len(workers) < 2:
                return ()
            keys = set()
            for source, target in zip(workers, workers[1:]):
                keys.update(resource.identity for resource in transfer_resources(
                    self.node_by_worker[source], self.node_by_worker[target]))
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
        if self.cluster_config.af.transfer_topology_mode == "physical":
            capacities = {key: value for key, value in capacities.items() if not key.startswith("link:")}
            capacities.update({key: 1 for key in self.domain_resources})
        return capacities

    def host_resources_for(self, physical_identities):
        scope = self.cluster_config.af.communication_timing.host_resource_scope
        keys = {"shared"} if scope == "shared" else {
            str(self.worker_id_for_physical(identity)) for identity in physical_identities
        }
        return tuple((f"host:{key}", self.host_submission_slots[key]) for key in sorted(keys))

    @staticmethod
    def format_link_resource(key: tuple[str, str]) -> str:
        if isinstance(key, str):
            return key
        return f"link:{key[0]}:{key[1]}"
