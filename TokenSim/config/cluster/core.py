"""Cluster config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import Field, dataclass

from TokenSim.config.af.domains import DomainRankMapping, ExecutionDomainConfig
from TokenSim.config.af.settings import AFConfig
from TokenSim.config.cluster.domains import ClusterDomainValidationMixin
from TokenSim.config.cluster.normalization import ClusterNormalizationMixin
from TokenSim.config.cluster.topology import ClusterTopologyMixin
from TokenSim.config.cluster.workers import WorkerGroupConfig
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


@dataclass(config=ConfigDict(extra="forbid"))
class ClusterConfig(
    ClusterNormalizationMixin, ClusterDomainValidationMixin, ClusterTopologyMixin
):
    num_workers: int
    networks: dict[str, str]
    worker_groups: list[WorkerGroupConfig] = Field(default_factory=list)
    kv_transfer: KVTransferConfig | None = None
    parallel_config: ParallelConfig | None = None
    # Optional explicit KV capacity.  A total capacity is divided across
    # data-parallel replicas; a per-rank value takes precedence only when the
    # total is omitted.
    kv_cache_capacity_tokens_total: int | None = None
    kv_cache_capacity_tokens_per_dp_rank: int | None = None
    # Optional catalog topology id (data/topologies/*.yaml). When omitted a
    # two-level topology is synthesized from ``networks``.
    topology: str | None = None
    af: AFConfig = Field(default_factory=AFConfig)

    @field_validator("num_workers", mode="before")
    @classmethod
    def _validate_num_workers_type(cls, value: Any) -> int:
        if type(value) is not int or value < 0:
            raise ConfigurationError(
                "cluster num_workers must be a non-negative integer"
            )
        return value

    def __post_init__(self) -> None:
        self._normalize_group_names()
        self._validate_groups()
        self.af.validate_dtype_style()
        if self.af.enabled:
            self._normalize_execution_domains()
            self._normalize_communication_fabrics()
            self._validate_af_topology()
            self._validate_execution_domains()
        self._validate_cache_capacity()

    def _normalize_group_names(self) -> None:
        role_indexes: dict[str, int] = {}
        for group in self.worker_groups:
            if group.name is None:
                role_name = group.role or "af"
                index = role_indexes.get(role_name, 0)
                group.name = f"legacy-{role_name}-{index}"
                role_indexes[role_name] = index + 1

    def _validate_groups(self) -> None:
        names: set[str] = set()
        for group in self.worker_groups:
            if not isinstance(group.name, str) or not group.name.strip():
                raise ConfigurationError("worker group name must be a non-empty string")
            if group.name in names:
                raise ConfigurationError(f"duplicate worker group name {group.name!r}")
            names.add(group.name)
            if group.network not in self.networks:
                raise ConfigurationError(
                    f"worker group {group.name!r} references unknown network "
                    + f"{group.network!r}"
                )

    def _validate_cache_capacity(self) -> None:
        integer(self.num_workers, "cluster num_workers")
        if sum(group.num_workers for group in self.worker_groups) != self.num_workers:
            raise ConfigurationError(
                "cluster num_workers must equal the worker group total"
            )
        for field_name in (
            "kv_cache_capacity_tokens_total",
            "kv_cache_capacity_tokens_per_dp_rank",
        ):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ConfigurationError(f"{field_name} must be positive when provided")
        if (
            self.kv_cache_capacity_tokens_total is not None
            and self.kv_cache_capacity_tokens_per_dp_rank is not None
        ):
            raise ConfigurationError(
                "kv_cache_capacity_tokens_total and "
                "kv_cache_capacity_tokens_per_dp_rank are mutually exclusive"
            )

    @classmethod
    def from_file(cls, filename):
        try:
            return cls(**json.loads(Path(filename).read_text()))
        except (OSError, ValueError, TypeError) as error:
            raise ConfigurationError(
                f"invalid cluster configuration {filename}: {error}"
            ) from error

    def effective_kv_transfer(
        self,
        override: KVTransferConfig | None = None,
    ) -> KVTransferConfig:
        return override or self.kv_transfer or KVTransferConfig.default()

    def effective_parallel_config(
        self,
        override: ParallelConfig | None = None,
    ) -> ParallelConfig:
        return override or self.parallel_config or ParallelConfig.default()

    def effective_kv_cache_capacity_per_dp_rank(
        self,
        parallel_config: ParallelConfig,
    ) -> int | None:
        if self.kv_cache_capacity_tokens_per_dp_rank is not None:
            return self.kv_cache_capacity_tokens_per_dp_rank
        if self.kv_cache_capacity_tokens_total is None:
            return None
        capacity = (
            self.kv_cache_capacity_tokens_total // parallel_config.data_parallel_size
        )
        if capacity == 0:
            raise ConfigurationError(
                "total KV capacity must provide tokens to every DP rank"
            )
        return capacity

    def workers(self, parallel_config: ParallelConfig | None = None):
        workers = [
            worker
            for worker_group in self.worker_groups
            for worker in worker_group.workers(self.networks)
        ]
        effective_config = (
            parallel_config or self.parallel_config or ParallelConfig.default()
        )
        if not self.af.enabled:
            validate_worker_count(len(workers), effective_config)
            for global_rank, worker in enumerate(workers):
                worker.rank_info = ParallelRankInfo.from_global_rank(
                    global_rank,
                    effective_config,
                )
            return workers

        mappings: dict[str, tuple[ExecutionDomainConfig, DomainRankMapping]] = {}
        for domain in sorted(self.af.execution_domains, key=lambda item: item.name):
            for instance in sorted(
                domain.logical_instances, key=lambda item: item.name
            ):
                for mapping in instance.rank_mapping:
                    mappings.setdefault(
                        mapping.physical_worker.identity,
                        (domain, mapping),
                    )
        for worker in workers:
            resolved = mappings.get(worker.physical_worker_ref.identity)
            if resolved is None:
                worker.rank_info = ParallelRankInfo()
                continue
            domain, mapping = resolved
            worker.rank_info = ParallelRankInfo(
                global_rank=mapping.domain_rank,
                rank_in_dp_group=(
                    mapping.pp_rank * domain.parallel.tensor_parallel_size
                    + mapping.tp_rank
                ),
                tp_rank=mapping.tp_rank,
                pp_rank=mapping.pp_rank,
                dp_rank=mapping.dp_rank,
                kv_cache_group_id=f"dp{mapping.dp_rank}-pp{mapping.pp_rank}",
            )
        return workers

    def __str__(self):
        return "".join(str(worker_group) for worker_group in self.worker_groups)


def validate_worker_count(num_workers: int, parallel_config: ParallelConfig) -> None:
    if parallel_config.world_size == 1:
        return
    if num_workers != parallel_config.world_size:
        raise ConfigurationError(
            "worker count must equal tensor_parallel_size * pipeline_parallel_size "
            + "* data_parallel_size for explicit parallel topologies; "
            + f"got num_workers={num_workers}, expected={parallel_config.world_size}"
        )
