from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.moe.config import MoEModelConfig


@dataclass
class ExpertPlacement:
    strategy: str
    ep_rank_count: int
    tensor_parallel_size: int
    num_experts: int
    rank_to_experts: dict[int, list[int]] = field(default_factory=dict)
    pp_stage_to_moe_layers: dict[int, list[int]] = field(default_factory=dict)

    def experts_for_rank(self, rank_info: ParallelRankInfo) -> list[int]:
        return self.rank_to_experts.get(self.ep_rank_for_rank_info(rank_info), [])

    def ep_rank_for_rank_info(self, rank_info: ParallelRankInfo) -> int:
        return rank_info.dp_rank * self.tensor_parallel_size + rank_info.tp_rank

    def moe_layers_for_pp_rank(self, pp_rank: int) -> list[int]:
        return self.pp_stage_to_moe_layers.get(pp_rank, [])

    def owns_moe_layers(self, pp_rank: int) -> bool:
        return bool(self.moe_layers_for_pp_rank(pp_rank))

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "ep_rank_count": self.ep_rank_count,
            "tensor_parallel_size": self.tensor_parallel_size,
            "num_experts": self.num_experts,
            "rank_to_experts": {
                str(rank): experts for rank, experts in sorted(self.rank_to_experts.items())
            },
            "pp_stage_to_moe_layers": {
                str(rank): layers
                for rank, layers in sorted(self.pp_stage_to_moe_layers.items())
            },
        }


def build_expert_placement(
    moe_config: MoEModelConfig,
    parallel_config: ParallelConfig,
    total_layers: int | None = None,
) -> ExpertPlacement:
    strategy = parallel_config.expert_placement_strategy or "linear"
    if strategy not in {"linear", "round_robin"}:
        raise ConfigurationError(
            "expert_placement_strategy must be 'linear' or 'round_robin'"
        )
    ep_rank_count = effective_ep_rank_count(parallel_config)
    rank_to_experts = _assign_experts(
        strategy=strategy,
        num_experts=moe_config.num_experts if moe_config.enabled else 0,
        ep_rank_count=ep_rank_count,
    )
    pp_stage_to_moe_layers = _assign_moe_layers(
        moe_config=moe_config,
        parallel_config=parallel_config,
        total_layers=total_layers,
    )
    return ExpertPlacement(
        strategy=strategy,
        ep_rank_count=ep_rank_count,
        tensor_parallel_size=parallel_config.tensor_parallel_size,
        num_experts=moe_config.num_experts if moe_config.enabled else 0,
        rank_to_experts=rank_to_experts,
        pp_stage_to_moe_layers=pp_stage_to_moe_layers,
    )


def effective_ep_rank_count(parallel_config: ParallelConfig) -> int:
    if not parallel_config.enable_expert_parallel:
        return 1
    return parallel_config.tensor_parallel_size * parallel_config.data_parallel_size


def rank_ep_rank(
    rank_info: ParallelRankInfo,
    parallel_config: ParallelConfig | None = None,
) -> int:
    tp_size = parallel_config.tensor_parallel_size if parallel_config else None
    if tp_size is None:
        raise ConfigurationError("parallel_config is required to compute EP rank")
    return rank_info.dp_rank * tp_size + rank_info.tp_rank


def _assign_experts(
    *,
    strategy: str,
    num_experts: int,
    ep_rank_count: int,
) -> dict[int, list[int]]:
    if num_experts <= 0:
        return {rank: [] for rank in range(ep_rank_count)}
    if ep_rank_count < 1:
        raise ConfigurationError("effective EP rank count must be at least 1")
    result = {rank: [] for rank in range(ep_rank_count)}
    if strategy == "linear":
        base = num_experts // ep_rank_count
        remainder = num_experts % ep_rank_count
        cursor = 0
        for rank in range(ep_rank_count):
            count = base + (1 if rank < remainder else 0)
            result[rank] = list(range(cursor, cursor + count))
            cursor += count
        return result
    if strategy == "round_robin":
        for expert_id in range(num_experts):
            result[expert_id % ep_rank_count].append(expert_id)
        return result
    raise ConfigurationError(f"unsupported expert placement strategy {strategy!r}")


def _assign_moe_layers(
    *,
    moe_config: MoEModelConfig,
    parallel_config: ParallelConfig,
    total_layers: int | None,
) -> dict[int, list[int]]:
    result = {rank: [] for rank in range(parallel_config.pipeline_parallel_size)}
    if not moe_config.enabled:
        return result
    for layer_id in moe_config.moe_layer_indices(total_layers):
        pp_rank = _pp_rank_for_layer(
            layer_id,
            total_layers or layer_id + 1,
            parallel_config.pipeline_parallel_size,
        )
        result[pp_rank].append(layer_id)
    return result


def _pp_rank_for_layer(layer_id: int, total_layers: int, pp_size: int) -> int:
    if pp_size <= 1:
        return 0
    cursor = 0
    for pp_rank in range(pp_size):
        count = total_layers // pp_size + (1 if pp_rank < total_layers % pp_size else 0)
        if cursor <= layer_id < cursor + count:
            return pp_rank
        cursor += count
    return pp_size - 1
