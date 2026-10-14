from dataclasses import asdict, replace
from typing import Any

from pydantic.dataclasses import dataclass

from TokenSim.errors import ConfigurationError

# Which ranks share one set of experts under expert parallelism.
#   global: every TP x DP rank of a pipeline stage forms one wide-EP group
#           (vLLM/SGLang wide EP: experts are spread over all replicas).
#   per_dp: each data-parallel replica keeps a full copy of the experts and
#           spreads them over its own TP ranks (InferenceX-style TP8/EP8 with
#           DP attention: 64 GPUs = 8 independent EP8 replicas).
EXPERT_PARALLEL_SCOPES = ("global", "per_dp")
EXPERT_PLACEMENT_STRATEGIES = ("linear", "round_robin")
ALL2ALL_BACKENDS = (
    "allgather_reducescatter",
    "naive",
    "deepep_high_throughput",
    "deepep_low_latency",
)


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    data_parallel_size_local: int = 1
    tensor_parallel_collective_latency_scale: float = 1.0
    pipeline_parallel_activation_latency_scale: float = 1.0
    enable_expert_parallel: bool = False
    expert_parallel_scope: str = "global"
    # Ranks per expert-parallel group. ``None`` means the whole scope
    # (tensor_parallel_size for ``per_dp``, tensor_parallel_size *
    # data_parallel_size for ``global``); an explicit value is validated
    # against the scope so a config can assert "EP8" and fail loudly when the
    # TP/DP layout would produce something else.
    expert_parallel_size: int | None = None
    expert_placement_strategy: str = "linear"
    all2all_backend: str = "allgather_reducescatter"

    def __post_init__(self) -> None:
        for field_name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "data_parallel_size_local",
        ):
            value = getattr(self, field_name)
            if value < 1:
                raise ConfigurationError(f"{field_name} must be at least 1")
        if self.data_parallel_rank < 0:
            raise ConfigurationError("data_parallel_rank must be non-negative")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise ConfigurationError(
                "data_parallel_rank must be smaller than data_parallel_size"
            )
        if self.data_parallel_size_local > self.data_parallel_size:
            raise ConfigurationError(
                "data_parallel_size_local cannot exceed data_parallel_size"
            )
        if self.expert_parallel_scope not in EXPERT_PARALLEL_SCOPES:
            raise ConfigurationError(
                f"expert_parallel_scope must be one of {list(EXPERT_PARALLEL_SCOPES)}"
            )
        if self.expert_placement_strategy not in EXPERT_PLACEMENT_STRATEGIES:
            raise ConfigurationError(
                f"expert_placement_strategy must be one of {list(EXPERT_PLACEMENT_STRATEGIES)}"
            )
        if self.all2all_backend not in ALL2ALL_BACKENDS:
            raise ConfigurationError(
                f"all2all_backend must be one of {list(ALL2ALL_BACKENDS)}"
            )
        self._validate_expert_parallel_size()

    def _validate_expert_parallel_size(self) -> None:
        size = self.expert_parallel_size
        if size is None:
            return
        if size < 1:
            raise ConfigurationError("expert_parallel_size must be at least 1")
        if not self.enable_expert_parallel:
            raise ConfigurationError(
                "expert_parallel_size requires enable_expert_parallel"
            )
        tp = self.tensor_parallel_size
        scope_ranks = self._expert_parallel_scope_ranks
        if self.expert_parallel_scope == "per_dp" and size != tp:
            raise ConfigurationError(
                "expert_parallel_scope 'per_dp' spreads the experts over the "
                f"tensor_parallel_size={tp} ranks of one data-parallel replica, so "
                f"expert_parallel_size must be {tp}, got {size}; MoE tensor "
                "parallelism inside an expert-parallel group is not modelled"
            )
        if size % tp != 0 or scope_ranks % size != 0:
            raise ConfigurationError(
                f"expert_parallel_size={size} must be a multiple of "
                f"tensor_parallel_size={tp} (groups hold whole data-parallel "
                f"replicas) and divide tensor_parallel_size * data_parallel_size="
                f"{scope_ranks}"
            )

    # -- rank counts ------------------------------------------------------------

    @property
    def ranks_per_dp_group(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def world_size(self) -> int:
        return self.ranks_per_dp_group * self.data_parallel_size

    @property
    def _expert_parallel_scope_ranks(self) -> int:
        if self.expert_parallel_scope == "per_dp":
            return self.tensor_parallel_size
        return self.tensor_parallel_size * self.data_parallel_size

    @property
    def expert_parallel_group_size(self) -> int:
        """Ranks that share one copy of the experts (1 without EP)."""
        if not self.enable_expert_parallel:
            return 1
        if self.expert_parallel_size is not None:
            return self.expert_parallel_size
        return self._expert_parallel_scope_ranks

    @property
    def expert_parallel_group_count(self) -> int:
        """Independent expert-parallel groups per pipeline stage."""
        return (
            self.tensor_parallel_size * self.data_parallel_size
        ) // self.expert_parallel_group_size

    @property
    def expert_parallel_replicas_per_group(self) -> int:
        """Data-parallel replicas whose tokens meet in one expert-parallel group.

        TP ranks of a replica see the same tokens, so a MoE layer inside a
        group receives ``replicas_per_group`` times one replica's tokens.
        """
        return self.expert_parallel_group_size // self.tensor_parallel_size

    # -- expert-parallel coordinates ------------------------------------------

    def expert_parallel_group(self, dp_rank: int, tp_rank: int) -> int:
        """Index of the expert-parallel group holding rank ``(dp_rank, tp_rank)``.

        Groups are contiguous in TP-fastest rank order: with ``per_dp`` the
        group index equals ``dp_rank``; with ``global`` (full scope) every
        rank of the stage is in group 0.
        """
        return (dp_rank * self.tensor_parallel_size + tp_rank) // self.expert_parallel_group_size

    def expert_parallel_rank(self, dp_rank: int, tp_rank: int) -> int:
        """Position of rank ``(dp_rank, tp_rank)`` inside its expert-parallel group."""
        return (dp_rank * self.tensor_parallel_size + tp_rank) % self.expert_parallel_group_size

    # -- construction ------------------------------------------------------------

    @classmethod
    def default(cls) -> "ParallelConfig":
        return cls()

    def override(
        self,
        *,
        tensor_parallel_size: int | None = None,
        pipeline_parallel_size: int | None = None,
        data_parallel_size: int | None = None,
        data_parallel_rank: int | None = None,
        data_parallel_size_local: int | None = None,
        enable_expert_parallel: bool | None = None,
        expert_parallel_scope: str | None = None,
        expert_parallel_size: int | None = None,
        expert_placement_strategy: str | None = None,
        all2all_backend: str | None = None,
    ) -> "ParallelConfig":
        """Copy with every non-``None`` argument replaced (CLI overrides)."""
        requested = {
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "data_parallel_size": data_parallel_size,
            "data_parallel_rank": data_parallel_rank,
            "data_parallel_size_local": data_parallel_size_local,
            "enable_expert_parallel": enable_expert_parallel,
            "expert_parallel_scope": expert_parallel_scope,
            "expert_parallel_size": expert_parallel_size,
            "expert_placement_strategy": expert_placement_strategy,
            "all2all_backend": all2all_backend,
        }
        return replace(self, **{k: v for k, v in requested.items() if v is not None})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParallelRankInfo:
    global_rank: int = 0
    rank_in_dp_group: int = 0
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    kv_cache_group_id: str = "dp0-pp0"

    @classmethod
    def from_global_rank(
        cls,
        global_rank: int,
        parallel_config: ParallelConfig,
    ) -> "ParallelRankInfo":
        if global_rank < 0:
            raise ConfigurationError("global rank must be non-negative")
        if global_rank >= parallel_config.world_size:
            return cls(
                global_rank=global_rank,
                rank_in_dp_group=global_rank,
                tp_rank=0,
                pp_rank=0,
                dp_rank=0,
                kv_cache_group_id="dp0-pp0",
            )
        rank_in_dp_group = global_rank % parallel_config.ranks_per_dp_group
        dp_rank = global_rank // parallel_config.ranks_per_dp_group
        pp_rank = rank_in_dp_group // parallel_config.tensor_parallel_size
        tp_rank = rank_in_dp_group % parallel_config.tensor_parallel_size
        return cls(
            global_rank=global_rank,
            rank_in_dp_group=rank_in_dp_group,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            kv_cache_group_id=f"dp{dp_rank}-pp{pp_rank}",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
