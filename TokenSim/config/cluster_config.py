import json
from pathlib import Path

from pydantic.dataclasses import dataclass, Field

from TokenSim.config.constants import WORKER_ROLE_PREFIXES, WORKER_ROLES
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError


@dataclass
class WorkerConfig:
    role: str
    hardware: str
    network: str
    nettype: str
    rank_info: ParallelRankInfo | None = None


@dataclass
class WorkerGroupConfig:
    role: str
    hardware: str
    num_workers: int
    network: str = "net1"

    def workers(self, networks):
        if self.role not in WORKER_ROLES:
            raise ConfigurationError(
                f"unsupported worker role {self.role!r}; expected one of "
                + f"{sorted(WORKER_ROLES)}"
            )
        return [
            WorkerConfig(self.role, self.hardware, self.network, networks[self.network])
            for _ in range(self.num_workers)
        ]

    def __str__(self):
        role_prefix = WORKER_ROLE_PREFIXES[self.role]
        return f"{role_prefix}{self.num_workers:g}"


@dataclass
class ClusterConfig:
    num_workers: int
    networks: dict[str, str]
    worker_groups: list[WorkerGroupConfig] = Field(default_factory=list)
    kv_transfer: KVTransferConfig | None = None
    parallel_config: ParallelConfig | None = None
    kv_cache_capacity_tokens_per_dp_rank: int | None = None
    kv_cache_capacity_tokens_total: int | None = None

    def __post_init__(self) -> None:
        if (
            self.kv_cache_capacity_tokens_per_dp_rank is not None
            and self.kv_cache_capacity_tokens_total is not None
        ):
            raise ConfigurationError(
                "set only one of kv_cache_capacity_tokens_per_dp_rank or "
                "kv_cache_capacity_tokens_total"
            )
        if (
            self.kv_cache_capacity_tokens_per_dp_rank is not None
            and self.kv_cache_capacity_tokens_per_dp_rank <= 0
        ):
            raise ConfigurationError(
                "kv_cache_capacity_tokens_per_dp_rank must be positive"
            )
        if (
            self.kv_cache_capacity_tokens_total is not None
            and self.kv_cache_capacity_tokens_total <= 0
        ):
            raise ConfigurationError("kv_cache_capacity_tokens_total must be positive")

    @classmethod
    def from_file(cls, filename):
        return cls(**json.loads(Path(filename).read_text()))

    def effective_kv_transfer(
        self,
        override: KVTransferConfig | None = None,
    ) -> KVTransferConfig:
        return override or self.kv_transfer or KVTransferConfig.default()

    def effective_parallel_config(
        self,
        model_parallel_config: ParallelConfig | None = None,
        override: ParallelConfig | None = None,
    ) -> ParallelConfig:
        return (
            override
            or self.parallel_config
            or model_parallel_config
            or ParallelConfig.default()
        )

    def effective_kv_cache_capacity_per_dp_rank(
        self,
        parallel_config: ParallelConfig,
    ) -> int | None:
        if self.kv_cache_capacity_tokens_per_dp_rank is not None:
            return self.kv_cache_capacity_tokens_per_dp_rank
        if self.kv_cache_capacity_tokens_total is None:
            return None
        return self.kv_cache_capacity_tokens_total // parallel_config.data_parallel_size

    def workers(self, parallel_config: ParallelConfig | None = None):
        workers = [
            worker
            for worker_group in self.worker_groups
            for worker in worker_group.workers(self.networks)
        ]
        effective_config = parallel_config or self.parallel_config or ParallelConfig.default()
        validate_worker_count(len(workers), effective_config)
        for global_rank, worker in enumerate(workers):
            worker.rank_info = ParallelRankInfo.from_global_rank(
                global_rank,
                effective_config,
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
