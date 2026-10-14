"""Runtime cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TokenSim.af.runtime_plan import AFRuntimePlan

from dataclasses import dataclass

from TokenSim.af.runtime_records import AFKVHeadPartition, AFReplicaRuntime
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


@dataclass
class AFCacheConfig:
    hybrid_layout = None

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)

    block_size: int
    model: str
    num_gpu_blocks: int
    num_cpu_blocks: int
    size_per_token: int
    size_per_token_unsharded: int
    model_param_size: int
    model_param_size_unsharded: int
    num_layers_per_rank: int
    total_num_layers: int
    num_kv_heads: int
    local_kv_heads: int
    head_dim: int
    rank_info: ParallelRankInfo

    @classmethod
    def from_plan(
        cls, plan: AFRuntimePlan, replica: AFReplicaRuntime
    ) -> "AFCacheConfig":
        if not replica.is_attention or replica.leader_rank_key is None:
            raise ConfigurationError(
                f"AF cache config requires an A replica, got {replica.domain!r}"
            )
        bindings = [plan.binding(*rank_key) for rank_key in replica.rank_keys]
        leader = plan.binding(*replica.leader_rank_key)
        partitions = [item.kv_partition for item in bindings]
        if any(partition is None for partition in partitions):
            raise ConfigurationError(
                f"AF A replica {replica.domain}:{replica.dp_rank} lacks KV partitions"
            )
        block_bytes = sum(
            partition.bytes_per_block for partition in partitions if partition
        )
        local_partition = leader.kv_partition
        model_bytes = plan.physical_state(leader.physical_identity).model_bytes
        return cls(
            block_size=plan.block_size,
            model=plan.model,
            num_gpu_blocks=replica.logical_gpu_blocks,
            num_cpu_blocks=replica.logical_cpu_blocks,
            size_per_token=block_bytes // plan.block_size,
            size_per_token_unsharded=block_bytes // plan.block_size,
            model_param_size=model_bytes,
            model_param_size_unsharded=sum(
                state.model_bytes for state in plan.physical_states
            ),
            num_layers_per_rank=plan.total_layers,
            total_num_layers=plan.total_layers,
            num_kv_heads=plan.num_kv_heads,
            local_kv_heads=local_partition.local_head_count,
            head_dim=plan.head_dim,
            rank_info=ParallelRankInfo(
                global_rank=leader.domain_rank,
                rank_in_dp_group=leader.tp_rank,
                tp_rank=leader.tp_rank,
                pp_rank=leader.pp_rank,
                dp_rank=leader.dp_rank,
                kv_cache_group_id=f"{leader.domain}:dp{leader.dp_rank}",
            ),
        )


def kv_head_partitions(
    *,
    num_kv_heads: int,
    tensor_parallel_size: int,
    head_dim: int,
    num_layers: int,
    block_size: int,
    dtype_size_bytes: int,
    mla_cache_width: int = 0,
) -> tuple[AFKVHeadPartition, ...]:
    for name, value in (
        ("num_kv_heads", num_kv_heads),
        ("tensor_parallel_size", tensor_parallel_size),
        ("head_dim", head_dim),
        ("num_layers", num_layers),
        ("block_size", block_size),
        ("dtype_size_bytes", dtype_size_bytes),
    ):
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"AF KV partition {name} must be positive")
    if mla_cache_width:
        if type(mla_cache_width) is not int or mla_cache_width < 1 or num_kv_heads != 1:
            raise ConfigurationError(
                "MLA cache requires positive latent+RoPE width and one shared cache"
            )
        token_bytes = mla_cache_width * dtype_size_bytes * num_layers
        return tuple(
            AFKVHeadPartition(
                tp_rank=rank,
                global_head_ids=(0,),
                replica_factor=tensor_parallel_size,
                bytes_per_token=token_bytes,
                bytes_per_block=token_bytes * block_size,
            )
            for rank in range(tensor_parallel_size)
        )
    if num_kv_heads >= tensor_parallel_size:
        if num_kv_heads % tensor_parallel_size:
            raise ConfigurationError(
                f"KV heads {num_kv_heads} cannot be evenly sharded across A-domain "
                f"TP={tensor_parallel_size}"
            )
        heads_per_rank = num_kv_heads // tensor_parallel_size
        assignments = tuple(
            tuple(range(rank * heads_per_rank, (rank + 1) * heads_per_rank))
            for rank in range(tensor_parallel_size)
        )
        replica_factor = 1
    else:
        if tensor_parallel_size % num_kv_heads:
            raise ConfigurationError(
                f"A-domain TP={tensor_parallel_size} cannot form unambiguous "
                f"replication groups for {num_kv_heads} KV heads"
            )
        replica_factor = tensor_parallel_size // num_kv_heads
        assignments = tuple(
            (rank // replica_factor,) for rank in range(tensor_parallel_size)
        )
    bytes_per_head_token = head_dim * 2 * dtype_size_bytes * num_layers
    return tuple(
        AFKVHeadPartition(
            tp_rank=rank,
            global_head_ids=head_ids,
            replica_factor=replica_factor,
            bytes_per_token=len(head_ids) * bytes_per_head_token,
            bytes_per_block=len(head_ids) * bytes_per_head_token * block_size,
        )
        for rank, head_ids in enumerate(assignments)
    )
