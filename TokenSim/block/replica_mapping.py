"""Replica mapping."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

from TokenSim.af.runtime import RankKey
from TokenSim.block.block import Device
from TokenSim.errors import SimulationStateError
from TokenSim.llm.llm_request import Request, RequestStatus


@dataclass(frozen=True)
class LocalKVShard:
    rank_key: RankKey
    device: Device
    block_number: int

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.rank_key[0],
            "domain_rank": self.rank_key[1],
            "device": self.device.name,
            "block_number": self.block_number,
        }


@dataclass(frozen=True)
class LogicalKVBlockMapping:
    request_id: int
    logical_block_index: int
    device: Device
    shards: tuple[LocalKVShard, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "logical_block_index": self.logical_block_index,
            "device": self.device.name,
            "shards": [shard.as_dict() for shard in self.shards],
        }


class ReplicaMappingMixin:
    def mappings_for_request(
        self, request_id: int
    ) -> tuple[LogicalKVBlockMapping, ...]:
        return tuple(
            self.logical_mappings[key]
            for key in sorted(self.logical_mappings)
            if key[0] == request_id
        )

    def validate_invariants(self, req: Request | None = None) -> None:
        request_ids = (req.id,) if req is not None else tuple(self._request_views)
        for request_id in request_ids:
            views = self._request_views.get(request_id)
            if views is None:
                raise SimulationStateError(
                    f"AF replica {self.replica_key} has no request owner state for "
                    f"request {request_id}"
                )
            counts = {
                rank_key: self.rank_managers[rank_key].block_table.get_num_blocks(
                    request_id
                )
                for rank_key in self.rank_keys
            }
            if len(set(counts.values())) != 1:
                raise SimulationStateError(
                    f"AF replica {self.replica_key} request {request_id} has "
                    f"missing KV shards: counts={counts}"
                )
            for index in range(next(iter(counts.values()), 0)):
                mapping = self.logical_mappings.get((request_id, index))
                if mapping is None:
                    raise SimulationStateError(
                        f"AF replica {self.replica_key} request {request_id} has "
                        f"orphan KV shards at logical block {index}"
                    )
                if tuple(shard.rank_key for shard in mapping.shards) != self.rank_keys:
                    raise SimulationStateError(
                        f"AF replica {self.replica_key} request {request_id} logical "
                        f"block {index} rank membership mismatch"
                    )
                ref_counts = []
                for shard in mapping.shards:
                    block = self.rank_managers[shard.rank_key].block_table.get_blocks(
                        request_id
                    )[index]
                    if (
                        block.block_number != shard.block_number
                        or block.device != shard.device
                    ):
                        raise SimulationStateError(
                            f"AF replica {self.replica_key} request {request_id} logical "
                            f"block {index} shard mismatch for rank={shard.rank_key}"
                        )
                    ref_counts.append(block.ref_count)
                if len(set(ref_counts)) != 1:
                    raise SimulationStateError(
                        f"AF replica {self.replica_key} request {request_id} logical "
                        f"block {index} reference mismatch: {ref_counts}"
                    )
            owner_req = views[self.leader_rank_key]
            self._validate_request_owner(owner_req, set_if_missing=False)
        for rank_key, manager in self.rank_managers.items():
            if (
                manager.gpu_allocator.get_num_allocated_blocks()
                > manager.num_total_gpu_blocks
            ):
                raise SimulationStateError(
                    f"AF replica {self.replica_key} rank={rank_key} exceeds GPU capacity"
                )
            if (
                manager.cpu_allocator.get_num_allocated_blocks()
                > manager.num_total_cpu_blocks
            ):
                raise SimulationStateError(
                    f"AF replica {self.replica_key} rank={rank_key} exceeds CPU capacity"
                )

    def _require_views(self, req: Request) -> dict[RankKey, Request]:
        views = self._request_views.get(req.id)
        if views is None:
            raise SimulationStateError(
                f"request {req.id} has no KV shards in AF replica {self.replica_key}"
            )
        for view in views.values():
            view._logical_tokens = req._logical_tokens
        return views

    def _view(self, rank_key: RankKey, req: Request, *, create: bool) -> Request:
        if rank_key == self.leader_rank_key:
            return req
        existing = self._request_views.get(req.id, {}).get(rank_key)
        if existing is not None:
            return existing
        view = copy.copy(req)
        view._physical_token_blocks = list(
            self.rank_managers[rank_key].block_table.get_blocks(req.id)
        )
        view.input_cache_keys = list(getattr(req, "input_cache_keys", []))
        view.af_a_owners = dict(getattr(req, "af_a_owners", {}))
        if create:
            return view
        return view

    def _rebuild_request_mappings(self, request_id: int) -> None:
        rank_blocks = {
            rank_key: self.rank_managers[rank_key].block_table.get_blocks(request_id)
            for rank_key in self.rank_keys
        }
        counts = {rank_key: len(blocks) for rank_key, blocks in rank_blocks.items()}
        if len(set(counts.values())) != 1:
            raise SimulationStateError(
                f"AF replica {self.replica_key} request {request_id} cannot build "
                f"logical mapping from rank counts {counts}"
            )
        for key in tuple(self.logical_mappings):
            if key[0] == request_id:
                self.logical_mappings.pop(key)
        for index in range(next(iter(counts.values()), 0)):
            blocks = tuple(rank_blocks[rank_key][index] for rank_key in self.rank_keys)
            devices = {block.device for block in blocks}
            if len(devices) != 1:
                raise SimulationStateError(
                    f"AF replica {self.replica_key} request {request_id} logical "
                    f"block {index} spans devices {devices}"
                )
            device = next(iter(devices))
            self.logical_mappings[(request_id, index)] = LogicalKVBlockMapping(
                request_id=request_id,
                logical_block_index=index,
                device=device,
                shards=tuple(
                    LocalKVShard(
                        rank_key=rank_key,
                        device=rank_blocks[rank_key][index].device,
                        block_number=rank_blocks[rank_key][index].block_number,
                    )
                    for rank_key in self.rank_keys
                ),
            )

    def _validate_request_owner(self, req: Request, *, set_if_missing: bool) -> None:
        owners = getattr(req, "af_a_owners", None)
        if not isinstance(owners, dict):
            return
        for domain, instance in self.owner_instances.items():
            existing = owners.get(domain)
            if existing is not None and existing != instance:
                transferred = (
                    getattr(req, "status", None) == RequestStatus.WAITING_FOR_KV
                )
                if not transferred:
                    raise SimulationStateError(
                        f"request {req.id} owner conflict in domain={domain}: "
                        f"existing={existing}, requested={instance} without KV transfer"
                    )
            if set_if_missing:
                owners[domain] = instance

    @staticmethod
    def _snapshot_request(req: Request) -> dict[str, object]:
        fields = (
            "cached_prefill_blocks",
            "cached_prefill_tokens",
            "effective_prefill_tokens",
            "reuse_hit_blocks",
            "reuse_miss_blocks",
            "input_cache_committed",
        )
        result = {field: getattr(req, field) for field in fields}
        result["input_cache_keys"] = list(req.input_cache_keys)
        result["physical_blocks"] = list(req._physical_token_blocks)
        return result

    @staticmethod
    def _restore_request(req: Request, snapshot: Mapping[str, object]) -> None:
        for field, value in snapshot.items():
            if field in {"input_cache_keys", "physical_blocks"}:
                continue
            setattr(req, field, value)
        req.input_cache_keys = list(snapshot["input_cache_keys"])
        req._physical_token_blocks = list(snapshot["physical_blocks"])
