"""Replica kv."""

from __future__ import annotations

from typing import Mapping

from TokenSim.af.runtime.records import AFReplicaRuntime
from TokenSim.af.runtime.plan import AFRuntimePlan
from TokenSim.af.runtime.records import RankKey
from TokenSim.block.block_manager import BlockManager
from TokenSim.block.replica_mapping import (
    LocalKVShard,
    LogicalKVBlockMapping,
    ReplicaMappingMixin,
)
from TokenSim.errors import ConfigurationError, OutOfBlocksError, SimulationStateError
from TokenSim.llm.llm_request import Request


class ReplicaKVBlockManager(ReplicaMappingMixin):
    """One scheduler-visible KV cache backed by one local pool per A TP rank."""

    grouped_cache = False

    def __init__(
        self,
        *,
        block_size: int,
        model: str,
        replica: AFReplicaRuntime,
        gpu_blocks_by_rank: Mapping[RankKey, int] | None = None,
        cpu_blocks_by_rank: Mapping[RankKey, int] | None = None,
        rank_key_aliases: Mapping[RankKey, RankKey] | None = None,
        owner_instances: Mapping[str, str] | None = None,
    ) -> None:
        if not replica.is_attention or replica.leader_rank_key is None:
            raise ConfigurationError(
                f"replica KV manager requires an A replica, got {replica.domain!r}"
            )
        self.block_size = block_size
        self.model = model
        self.replica = replica
        self.replica_key = replica.replica_key
        self.domain = replica.domain
        self.instance = replica.instance
        self.dp_rank = replica.dp_rank
        self.rank_keys = tuple(replica.rank_keys)
        self.leader_rank_key = replica.leader_rank_key
        self.rank_key_aliases = dict(rank_key_aliases or {})
        self.owner_instances = dict(
            owner_instances or {replica.domain: replica.instance}
        )
        gpu_capacities = dict(gpu_blocks_by_rank or replica.local_gpu_blocks)
        cpu_capacities = dict(cpu_blocks_by_rank or replica.local_cpu_blocks)
        if set(gpu_capacities) != set(self.rank_keys):
            raise ConfigurationError(
                f"AF replica {self.replica_key} GPU capacities do not match ranks: "
                f"expected={self.rank_keys}, actual={tuple(gpu_capacities)}"
            )
        if set(cpu_capacities) != set(self.rank_keys):
            raise ConfigurationError(
                f"AF replica {self.replica_key} CPU capacities do not match ranks: "
                f"expected={self.rank_keys}, actual={tuple(cpu_capacities)}"
            )
        self.rank_managers = {
            rank_key: BlockManager(
                block_size=block_size,
                num_gpu_blocks=gpu_capacities[rank_key],
                num_cpu_blocks=cpu_capacities[rank_key],
                model=model,
                watermark=0,
            )
            for rank_key in self.rank_keys
        }
        self.num_total_gpu_blocks = min(gpu_capacities.values())
        self.num_total_cpu_blocks = min(cpu_capacities.values())
        self.watermark_blocks = 0
        leader = self.rank_managers[self.leader_rank_key]
        self.block_table = leader.block_table
        self.kv_cache_manager = leader.kv_cache_manager
        self.gpu_allocator = leader.gpu_allocator
        self.cpu_allocator = leader.cpu_allocator
        self._request_views: dict[int, dict[RankKey, Request]] = {}
        self.logical_mappings: dict[tuple[int, int], LogicalKVBlockMapping] = {}

    @classmethod
    def from_plan(
        cls,
        plan: AFRuntimePlan,
        replica: AFReplicaRuntime,
        aliases: tuple[AFReplicaRuntime, ...] = (),
    ) -> "ReplicaKVBlockManager":
        canonical_by_physical = {
            plan.binding(*rank_key).physical_identity: rank_key
            for rank_key in replica.rank_keys
        }
        rank_key_aliases: dict[RankKey, RankKey] = {}
        owner_instances = {replica.domain: replica.instance}
        for alias in aliases:
            owner_instances[alias.domain] = alias.instance
            for rank_key in alias.rank_keys:
                physical = plan.binding(*rank_key).physical_identity
                try:
                    rank_key_aliases[rank_key] = canonical_by_physical[physical]
                except KeyError as exc:
                    raise ConfigurationError(
                        f"AF replica alias {alias.replica_key} uses physical worker "
                        f"{physical!r} outside canonical replica {replica.replica_key}"
                    ) from exc
        return cls(
            block_size=plan.block_size,
            model=plan.model,
            replica=replica,
            rank_key_aliases=rank_key_aliases,
            owner_instances=owner_instances,
        )

    def can_allocate(self, req: Request) -> bool:
        self._validate_request_owner(req, set_if_missing=False)
        return all(
            manager.can_allocate(self._view(rank_key, req, create=False))
            for rank_key, manager in self.rank_managers.items()
        )

    def allocate(self, req: Request) -> None:
        self._validate_request_owner(req, set_if_missing=False)
        if req.id in self._request_views:
            raise SimulationStateError(
                f"request {req.id} already has KV shards in AF replica "
                f"{self.replica_key}"
            )
        views = {
            rank_key: self._view(rank_key, req, create=True)
            for rank_key in self.rank_keys
        }
        if not all(
            self.rank_managers[rank_key].can_allocate(views[rank_key])
            for rank_key in self.rank_keys
        ):
            raise OutOfBlocksError(
                f"AF replica {self.replica_key} cannot allocate request {req.id} "
                "on every TP rank"
            )
        snapshot = self._snapshot_request(req)
        completed: list[RankKey] = []
        try:
            for rank_key in self.rank_keys:
                self.rank_managers[rank_key].allocate(views[rank_key])
                completed.append(rank_key)
        except Exception:
            for rank_key in reversed(completed):
                self.rank_managers[rank_key].release_request_blocks(views[rank_key])
            self._restore_request(req, snapshot)
            raise
        self._request_views[req.id] = views
        self._validate_request_owner(req, set_if_missing=True)
        self._rebuild_request_mappings(req.id)
        self.validate_invariants(req)

    def can_append_slot(self, req: Request, reserved_blocks: int = 0) -> bool:
        views = self._require_views(req)
        return all(
            self.rank_managers[rank_key].can_append_slot(
                views[rank_key], reserved_blocks
            )
            for rank_key in self.rank_keys
        )

    def append_slot(self, req: Request) -> None:
        views = self._require_views(req)
        if not self.can_append_slot(req):
            raise OutOfBlocksError(
                f"AF replica {self.replica_key} cannot append request {req.id} "
                "on every TP rank"
            )
        old_counts = {
            rank_key: manager.block_table.get_num_blocks(req.id)
            for rank_key, manager in self.rank_managers.items()
        }
        completed: list[RankKey] = []
        try:
            for rank_key in self.rank_keys:
                self.rank_managers[rank_key].append_slot(views[rank_key])
                completed.append(rank_key)
        except Exception:
            for rank_key in reversed(completed):
                manager = self.rank_managers[rank_key]
                if manager.block_table.get_num_blocks(req.id) > old_counts[rank_key]:
                    manager.rollback_last_block(views[rank_key])
            raise
        self._rebuild_request_mappings(req.id)
        self.validate_invariants(req)

    def fork(self, parent: Request, child: Request) -> None:
        parent_views = self._require_views(parent)
        if child.id in self._request_views:
            raise SimulationStateError(
                f"fork child request {child.id} already has AF KV shards"
            )
        child_views = {
            rank_key: self._view(rank_key, child, create=True)
            for rank_key in self.rank_keys
        }
        completed: list[RankKey] = []
        try:
            for rank_key in self.rank_keys:
                self.rank_managers[rank_key].fork(
                    parent_views[rank_key], child_views[rank_key]
                )
                completed.append(rank_key)
        except Exception:
            for rank_key in reversed(completed):
                self.rank_managers[rank_key].release_request_blocks(
                    child_views[rank_key]
                )
            raise
        self._request_views[child.id] = child_views
        self._validate_request_owner(child, set_if_missing=True)
        self._rebuild_request_mappings(child.id)
        self.validate_invariants(parent)
        self.validate_invariants(child)

    def swap_out(self, req: Request) -> None:
        self._swap(req, to_gpu=False)

    def swap_in(self, req: Request) -> None:
        self._swap(req, to_gpu=True)

    def _swap(self, req: Request, *, to_gpu: bool) -> None:
        views = self._require_views(req)
        can_swap = "can_swap_in" if to_gpu else "can_swap_out"
        operation = "swap_in" if to_gpu else "swap_out"
        if not all(
            getattr(self.rank_managers[rank_key], can_swap)(views[rank_key])
            for rank_key in self.rank_keys
        ):
            raise OutOfBlocksError(
                f"AF replica {self.replica_key} cannot {operation} request "
                f"{req.id} on every TP rank"
            )
        completed: list[RankKey] = []
        try:
            for rank_key in self.rank_keys:
                getattr(self.rank_managers[rank_key], operation)(views[rank_key])
                completed.append(rank_key)
        except Exception:
            inverse = "swap_out" if to_gpu else "swap_in"
            for rank_key in reversed(completed):
                getattr(self.rank_managers[rank_key], inverse)(views[rank_key])
            self._rebuild_request_mappings(req.id)
            self.validate_invariants(req)
            raise
        self._rebuild_request_mappings(req.id)
        self.validate_invariants(req)

    def commit_input_cache(self, req: Request) -> None:
        views = self._require_views(req)
        for rank_key in self.rank_keys:
            self.rank_managers[rank_key].commit_input_cache(views[rank_key])

    def commit_finished_cache(self, req: Request) -> None:
        views = self._require_views(req)
        for rank_key in self.rank_keys:
            self.rank_managers[rank_key].commit_finished_cache(views[rank_key])

    def free(self, req: Request) -> None:
        self.commit_finished_cache(req)
        self.release_request_blocks(req)

    def release_request_blocks(self, req: Request) -> None:
        views = self._require_views(req)
        self.validate_invariants(req)
        for rank_key in reversed(self.rank_keys):
            self.rank_managers[rank_key].release_request_blocks(views[rank_key])
        self._request_views.pop(req.id, None)
        for key in tuple(self.logical_mappings):
            if key[0] == req.id:
                self.logical_mappings.pop(key)

    def reserve_gpu_blocks(self, num_blocks: int) -> None:
        completed: list[RankKey] = []
        try:
            for rank_key in self.rank_keys:
                self.rank_managers[rank_key].reserve_gpu_blocks(num_blocks)
                completed.append(rank_key)
        except Exception:
            for rank_key in reversed(completed):
                self.rank_managers[rank_key].release_reserved_blocks(num_blocks)
            raise

    def release_reserved_blocks(self, num_blocks: int | None = None) -> None:
        for rank_key in reversed(self.rank_keys):
            self.rank_managers[rank_key].release_reserved_blocks(num_blocks)

    def get_num_blocks(self, req: Request) -> int:
        return self.block_table.get_num_blocks(req.id) or req.num_physical_token_blocks

    def get_request_block_counts(self, requests: list[Request]) -> dict[int, int]:
        return {req.id: self.get_num_blocks(req) for req in requests}

    def get_gpu_status(self) -> tuple[int, int, int]:
        free = min(
            manager.gpu_allocator.get_num_free_blocks()
            for manager in self.rank_managers.values()
        )
        total = self.num_total_gpu_blocks
        return free, total - free, total

    def get_cpu_status(self) -> tuple[int, int, int]:
        free = min(
            manager.cpu_allocator.get_num_free_blocks()
            for manager in self.rank_managers.values()
        )
        total = self.num_total_cpu_blocks
        return free, total - free, total

    def rank_status(self, rank_key: RankKey) -> dict[str, int]:
        rank_key = self.rank_key_aliases.get(rank_key, rank_key)
        try:
            manager = self.rank_managers[rank_key]
        except KeyError as exc:
            raise ConfigurationError(
                f"AF replica {self.replica_key} has no rank {rank_key}"
            ) from exc
        _, gpu_used, gpu_total = manager.get_gpu_status()
        _, cpu_used, cpu_total = manager.get_cpu_status()
        return {
            "gpu_capacity": gpu_total,
            "gpu_used": gpu_used,
            "cpu_capacity": cpu_total,
            "cpu_used": cpu_used,
        }
