from __future__ import annotations

import simpy

from TokenSim.config.cluster_config import AFLinkConfig, ClusterConfig, WorkerConfig
from TokenSim.errors import ConfigurationError


class AFResourceRegistry:
    def __init__(
        self,
        env: simpy.Environment,
        cluster_config: ClusterConfig,
        worker_configs: list[WorkerConfig],
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
    ) -> tuple[tuple[tuple[str, str], simpy.Resource], ...]:
        group = resource_group or link.resource_group
        if link.duplex == "half":
            suffix = "shared"
        else:
            suffix = {
                "a_to_f": "a_to_f",
                "f_to_a": "f_to_a",
                "tp_collective": "a_to_f",
                "ep_dispatch": "a_to_f",
                "ep_combine": "f_to_a",
            }.get(direction, direction)
        key = (group, suffix)
        resource = self.communication_resources.get(key)
        if resource is None:
            raise ConfigurationError(
                f"AF communication resource {key!r} is missing; domain-internal "
                "fabric must alias a configured resource_group"
            )
        return ((key, resource),)

    def resource_capacities(self) -> dict[str, int]:
        capacities = {f"worker:{worker_id}": 1 for worker_id in self.compute_slots}
        capacities.update(
            {
                self.format_link_resource(key): resource.capacity
                for key, resource in self.communication_resources.items()
            }
        )
        return capacities

    @staticmethod
    def format_link_resource(key: tuple[str, str]) -> str:
        return f"link:{key[0]}:{key[1]}"
