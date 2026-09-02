from __future__ import annotations

from dataclasses import dataclass

import simpy

from TokenSim.config.cluster_config import AFLinkConfig, ClusterConfig, WorkerConfig
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True, order=True)
class AFWorkerToken:
    worker_id: int
    group_name: str
    hardware: str


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

        self.ffn_pools: dict[str, simpy.Store] = {}
        for group in cluster_config.worker_groups:
            if "af_ffn" not in group.capabilities:
                continue
            worker_ids = self.group_worker_ids[group.name]
            pool = simpy.Store(env, capacity=len(worker_ids))
            for worker_id in worker_ids:
                worker = self.worker_configs[worker_id]
                pool.put(AFWorkerToken(worker_id, group.name, worker.hardware))
            self.ffn_pools[group.name] = pool

        self.communication_resources = {
            key: simpy.Resource(env, capacity=config.capacity)
            for key, config in cluster_config.af_communication_resources.items()
        }

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

    def first_group_worker(self, group_name: str) -> WorkerConfig:
        try:
            worker_id = self.group_worker_ids[group_name][0]
        except (KeyError, IndexError) as exc:
            raise ConfigurationError(
                f"AF worker group {group_name!r} has no workers"
            ) from exc
        return self.worker(worker_id)

    def ffn_pool(self, group_name: str) -> simpy.Store:
        try:
            return self.ffn_pools[group_name]
        except KeyError as exc:
            raise ConfigurationError(
                f"AF FFN group {group_name!r} has no compute pool"
            ) from exc

    def communication_resource(
        self,
        link: AFLinkConfig,
        direction: str,
    ) -> tuple[tuple[str, str], simpy.Resource]:
        suffix = "shared" if link.duplex == "half" else direction
        key = (link.resource_group, suffix)
        try:
            return key, self.communication_resources[key]
        except KeyError as exc:
            raise ConfigurationError(
                f"AF communication resource {key!r} is missing"
            ) from exc

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
