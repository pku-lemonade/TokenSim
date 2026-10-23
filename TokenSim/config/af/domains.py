"""Execution domain."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, StrictInt
from pydantic.dataclasses import Field, dataclass

from TokenSim.config.af.common import _COLLECTIVE_OPERATIONS
from TokenSim.comm.collective_types import ALGORITHMS
from TokenSim.errors import ConfigurationError

KV_PARALLEL_LAYERS = ("full_attention", "all")


@dataclass(frozen=True)
class DomainExpertParallelView:
    mode: Literal["expert_tp", "rank_reuse", "expert_sharded"]
    expert_parallel_size: int
    expert_tensor_parallel_size: int
    base_world_size: int
    base_tensor_parallel_size: int

    def ep_rank(self, *, dp_rank: int, tp_rank: int) -> int:
        data_parallel_size = self.base_world_size // self.base_tensor_parallel_size
        if not 0 <= dp_rank < data_parallel_size:
            raise ConfigurationError(
                f"dp_rank={dp_rank} is outside rank-reuse DP size "
                f"{data_parallel_size}"
            )
        if not 0 <= tp_rank < self.base_tensor_parallel_size:
            raise ConfigurationError(
                f"tp_rank={tp_rank} is outside rank-reuse TP size "
                f"{self.base_tensor_parallel_size}"
            )
        if self.mode == "expert_tp":
            return 0
        return (
            dp_rank * self.base_tensor_parallel_size + tp_rank
        ) // self.expert_tensor_parallel_size


@dataclass(frozen=True)
class PhysicalWorkerRef:
    group_name: str
    worker_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.group_name, str) or not self.group_name.strip():
            raise ConfigurationError(
                "physical worker group_name must be a non-empty string"
            )
        if type(self.worker_index) is not int or self.worker_index < 0:
            raise ConfigurationError(
                "physical worker worker_index must be a non-negative integer"
            )

    @property
    def identity(self) -> str:
        return f"{self.group_name}:{self.worker_index}"


@dataclass(frozen=True)
class DomainParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_placement_strategy: str = "linear"
    pipeline_layer_counts: tuple[StrictInt, ...] | None = None
    kv_parallel_size: int = 1
    kv_parallel_layers: str = "full_attention"
    kv_append_block: int = 16

    def __post_init__(self) -> None:
        for field_name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "expert_parallel_size",
            "kv_parallel_size",
            "kv_append_block",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 1:
                raise ConfigurationError(
                    f"execution domain {field_name} must be a positive integer"
                )
        if self.expert_placement_strategy not in {"linear", "round_robin"}:
            raise ConfigurationError(
                "execution domain expert_placement_strategy must be 'linear' or "
                "'round_robin'"
            )
        if self.pipeline_layer_counts is not None and (
            len(self.pipeline_layer_counts) != self.pipeline_parallel_size
            or any(count < 1 for count in self.pipeline_layer_counts)
        ):
            raise ConfigurationError(
                "pipeline_layer_counts needs one positive layer count per PP stage"
            )
        if self.tensor_parallel_size % self.kv_parallel_size:
            raise ConfigurationError(
                "execution domain kv_parallel_size must divide tensor_parallel_size"
            )
        if self.kv_parallel_layers not in KV_PARALLEL_LAYERS:
            raise ConfigurationError(
                f"execution domain kv_parallel_layers must be one of "
                f"{KV_PARALLEL_LAYERS}"
            )

    @property
    def ranks_per_dp_group(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def world_size(self) -> int:
        return self.ranks_per_dp_group * self.data_parallel_size

    def ranks_per_replica(self, capability: str) -> int:
        _ = capability
        return self.ranks_per_dp_group

    def required_rank_count(self, capability: str) -> int:
        _ = capability
        return self.world_size

    def expert_parallel_view(
        self, capability: str, *, context: str = "execution domain"
    ) -> DomainExpertParallelView:
        if capability != "af_ffn":
            if self.expert_parallel_size != 1:
                raise ConfigurationError(
                    f"{context} expert_parallel_size is only valid for af_ffn"
                )
            return DomainExpertParallelView(
                mode="expert_tp",
                expert_parallel_size=1,
                expert_tensor_parallel_size=self.tensor_parallel_size,
                base_world_size=self.tensor_parallel_size * self.data_parallel_size,
                base_tensor_parallel_size=self.tensor_parallel_size,
            )
        world = self.tensor_parallel_size * self.data_parallel_size
        expert_tp = world // self.expert_parallel_size
        if self.expert_parallel_size != 1 and (
            world % self.expert_parallel_size
            or not expert_tp
            or self.tensor_parallel_size % expert_tp
        ):
            raise ConfigurationError(
                f"{context} expert_parallel_size={self.expert_parallel_size} must divide "
                f"TP*DP={world} and its Expert TP must divide base TP={self.tensor_parallel_size}"
            )
        rank_reuse = self.expert_parallel_size == world and world > 1
        return DomainExpertParallelView(
            mode=(
                "rank_reuse"
                if rank_reuse
                else "expert_tp" if self.expert_parallel_size == 1 else "expert_sharded"
            ),
            expert_parallel_size=self.expert_parallel_size,
            expert_tensor_parallel_size=(
                self.tensor_parallel_size
                if self.expert_parallel_size == 1
                else expert_tp
            ),
            base_world_size=world,
            base_tensor_parallel_size=self.tensor_parallel_size,
        )


@dataclass(frozen=True)
class DomainRankMapping:
    domain_rank: int
    tp_rank: int
    pp_rank: int
    dp_rank: int
    physical_worker: PhysicalWorkerRef
    ep_rank: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "domain_rank",
            "tp_rank",
            "pp_rank",
            "dp_rank",
            "ep_rank",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ConfigurationError(
                    f"domain rank mapping {field_name} must be a non-negative integer"
                )


@dataclass
class LogicalInstanceConfig:
    name: str
    rank_mapping: list[DomainRankMapping] = Field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("logical instance name must be a non-empty string")
        if not self.rank_mapping:
            raise ConfigurationError(
                f"logical instance {self.name!r} rank_mapping must be non-empty"
            )


@dataclass(frozen=True)
class DomainCollectiveConfig:
    operation: str
    after_stage: str
    participants: tuple[int, ...]
    resource_group: str | None = None
    resource_lane: str | None = None
    network: str | None = None
    payload_bytes: int | None = None
    algorithm: str = "ring"

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ConfigurationError(
                f"domain collective algorithm must be one of {ALGORITHMS}"
            )
        if self.operation not in _COLLECTIVE_OPERATIONS:
            raise ConfigurationError(
                "domain collective operation must be one of "
                f"{sorted(_COLLECTIVE_OPERATIONS)}; got {self.operation!r}"
            )
        if not isinstance(self.after_stage, str) or not self.after_stage.strip():
            raise ConfigurationError(
                f"domain collective {self.operation!r} after_stage must be a "
                "non-empty string"
            )
        if not self.participants or any(
            type(rank) is not int or rank < 0 for rank in self.participants
        ):
            raise ConfigurationError(
                f"domain collective {self.operation!r} participants must be "
                "non-empty non-negative replica-local tp_rank offsets"
            )
        if len(self.participants) != len(set(self.participants)):
            raise ConfigurationError(
                f"domain collective {self.operation!r} participants contains duplicates"
            )
        for field_name in ("resource_group", "resource_lane", "network"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigurationError(
                    f"domain collective {self.operation!r} {field_name} must be a "
                    "non-empty string when provided"
                )
        if self.payload_bytes is not None and (
            type(self.payload_bytes) is not int or self.payload_bytes < 0
        ):
            raise ConfigurationError(
                f"domain collective {self.operation!r} payload_bytes must be "
                "a non-negative integer"
            )


@dataclass(config=ConfigDict(extra="forbid"))
class ExecutionDomainConfig:
    name: str
    capability: str
    worker_group: str
    parallel: DomainParallelConfig = Field(default_factory=DomainParallelConfig)
    logical_instances: list[LogicalInstanceConfig] = Field(default_factory=list)
    communication_plan: list[DomainCollectiveConfig] = Field(default_factory=list)

    @property
    def ranks_per_replica(self) -> int:
        return self.parallel.ranks_per_replica(self.capability)

    @property
    def required_rank_count(self) -> int:
        return self.parallel.required_rank_count(self.capability)

    @property
    def expert_parallel(self) -> DomainExpertParallelView:
        return self.parallel.expert_parallel_view(
            self.capability, context=f"execution domain {self.name!r}"
        )

    def resolve_collective_participants(
        self, collective: DomainCollectiveConfig, dp_rank: int, pp_rank: int = 0
    ) -> tuple[int, ...]:
        instance = next(
            (
                item
                for item in self.logical_instances
                if {mapping.dp_rank for mapping in item.rank_mapping} == {dp_rank}
            ),
            None,
        )
        if instance is None:
            raise ConfigurationError(
                f"execution domain {self.name!r} has no DP replica {dp_rank}"
            )
        mapping_by_tp = {
            mapping.tp_rank: mapping.domain_rank
            for mapping in instance.rank_mapping
            if mapping.pp_rank == pp_rank
        }
        missing = set(collective.participants) - set(mapping_by_tp)
        if missing:
            raise ConfigurationError(
                f"execution domain {self.name!r} collective {collective.operation!r} "
                f"has unresolved replica-local tp_rank participants {sorted(missing)} "
                f"for dp_rank={dp_rank}"
            )
        return tuple(mapping_by_tp[rank] for rank in collective.participants)

    def __post_init__(self) -> None:
        for field_name in ("name", "worker_group"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"execution domain field {field_name} must be a non-empty string"
                )
        if self.capability not in {"af_attention", "af_ffn"}:
            raise ConfigurationError(
                f"execution domain {self.name!r} has unsupported capability "
                f"{self.capability!r}"
            )
        if self.capability != "af_attention" and self.parallel.kv_parallel_size != 1:
            raise ConfigurationError(
                f"execution domain {self.name!r} declares kv_parallel_size for "
                "a non-attention capability"
            )
        instance_names = [instance.name for instance in self.logical_instances]
        if len(instance_names) != len(set(instance_names)):
            raise ConfigurationError(
                f"execution domain {self.name!r} has duplicate logical instance names"
            )
