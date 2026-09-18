import json
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator
from pydantic.dataclasses import dataclass, Field

from TokenSim.config.constants import (
    AF_PHASE_ORDER,
    AF_PHASES,
    LEGACY_ROLE_CAPABILITIES,
    WORKER_CAPABILITIES,
    WORKER_CAPABILITY_ORDER,
    WORKER_ROLE_PREFIXES,
    WORKER_ROLES,
)
from TokenSim.config.kv_transfer_config import KVTransferConfig
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError


@dataclass
class WorkerConfig:
    role: str | None
    hardware: str
    network: str
    nettype: str
    name: str | None = None
    capabilities: tuple[str, ...] = ()
    rank_info: ParallelRankInfo | None = None
    physical_worker_ref: "PhysicalWorkerRef | None" = None
    # Explicit topology device index (optional) and the operator-data backend
    # (e.g. vllm, trtllm, analytical) whose tables price this worker's kernels.
    device_index: int | None = None
    operator_backend: str | None = None


@dataclass
class WorkerGroupConfig:
    role: str | None = None
    hardware: str = ""
    num_workers: int = 0
    network: str = "net1"
    name: str | None = None
    capabilities: tuple[str, ...] = ()
    # Explicit topology device indices for the group's workers (optional).
    device_indices: list[int] | None = None
    # Which operator-data backend (e.g. trtllm, vllm, analytical) to use.
    operator_backend: str | None = None

    @field_validator("num_workers", mode="before")
    @classmethod
    def _validate_num_workers_type(cls, value: Any) -> int:
        if type(value) is not int:
            raise ConfigurationError(
                "worker group num_workers must be a positive integer"
            )
        return value

    def __post_init__(self) -> None:
        context = f"worker group {self.name!r}" if self.name else "worker group"
        if self.role is not None and self.role not in WORKER_ROLES:
            raise ConfigurationError(
                f"{context} has unsupported role {self.role!r}; expected one of "
                + f"{sorted(WORKER_ROLES)}"
            )
        if self.num_workers < 1:
            raise ConfigurationError(
                f"{context} num_workers must be a positive integer"
            )
        if not isinstance(self.hardware, str) or not self.hardware.strip():
            raise ConfigurationError(f"{context} hardware must be a non-empty string")
        if not isinstance(self.network, str) or not self.network.strip():
            raise ConfigurationError(f"{context} network must be a non-empty string")
        if self.device_indices is not None and len(self.device_indices) != self.num_workers:
            raise ConfigurationError(
                f"{context} device_indices must have num_workers entries"
            )
        if self.operator_backend is not None and (
            not isinstance(self.operator_backend, str) or not self.operator_backend.strip()
        ):
            raise ConfigurationError(f"{context} operator_backend must be a non-empty string")

        declared = tuple(self.capabilities)
        if len(declared) != len(set(declared)):
            raise ConfigurationError(
                f"{context} capabilities contains duplicate values"
            )
        unknown = set(declared) - WORKER_CAPABILITIES
        if unknown:
            raise ConfigurationError(
                f"{context} has unsupported capabilities {sorted(unknown)}; expected values "
                + f"from {sorted(WORKER_CAPABILITIES)}"
            )

        inferred = LEGACY_ROLE_CAPABILITIES.get(self.role, frozenset())
        if not declared:
            if self.role is None:
                raise ConfigurationError(
                    f"{context} must declare a legacy role or non-empty capabilities"
                )
            declared = tuple(inferred)
        elif self.role is not None:
            declared_complete = set(declared) & {"prefill", "decode_full"}
            if declared_complete != set(inferred):
                raise ConfigurationError(
                    f"{context} role {self.role!r} conflicts with capabilities "
                    + f"{sorted(declared_complete)}"
                )
        self.capabilities = tuple(
            capability
            for capability in WORKER_CAPABILITY_ORDER
            if capability in declared
        )

    def workers(self, networks):
        return [
            WorkerConfig(
                role=self.role,
                hardware=self.hardware,
                network=self.network,
                nettype=networks[self.network],
                name=self.name,
                capabilities=self.capabilities,
                physical_worker_ref=PhysicalWorkerRef(
                    group_name=self.name or f"unbound:{self.role or 'af'}",
                    worker_index=worker_index,
                ),
                device_index=(
                    self.device_indices[worker_index]
                    if self.device_indices is not None
                    else None
                ),
                operator_backend=self.operator_backend,
            )
            for worker_index in range(self.num_workers)
        ]

    def __str__(self):
        if self.role is None:
            return f"{self.name or 'af'}{self.num_workers:g}"
        return f"{WORKER_ROLE_PREFIXES[self.role]}{self.num_workers:g}"


@dataclass(frozen=True)
class AFCommunicationResourceConfig:
    resource_group: str
    resource_key: tuple[str, str]
    network: str
    duplex: str
    direction: str
    capacity: int
    queue_duration_field: str = "queue_duration"
    service_duration_field: str = "service_duration"


@dataclass(frozen=True)
class AFCommunicationLaneConfig:
    """Compatibility descriptor for legacy inline AF fabric declarations."""

    name: str
    capacity: int


@dataclass(frozen=True)
class AFCommunicationFabricConfig:
    """Compatibility container; runtime resources are still built from AF links."""

    resource_group: str
    network: str
    duplex: str
    lanes: tuple[AFCommunicationLaneConfig, ...]


@dataclass(frozen=True)
class AFEffectiveDTypes:
    weight_dtype: str = "fp16"
    activation_dtype: str = "fp16"
    kv_cache_dtype: str = "fp16"
    a_to_f_dtype: str = "fp16"
    f_to_a_dtype: str = "fp16"
    sources: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AFCommunicationTimingConfig:
    enabled: bool = False
    host_submit_us: float = 5.0
    host_completion_us: float = 0.0
    host_resource_scope: str = "per_worker"


@dataclass(frozen=True)
class DomainExpertParallelView:
    mode: str = "expert_tp"
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    base_world_size: int = 1
    base_tensor_parallel_size: int = 1


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
    # Legacy spelling accepted by existing AF JSON configurations.
    expert_parallel_mode: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "expert_parallel_size",
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

    @property
    def ranks_per_dp_group(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def world_size(self) -> int:
        return self.ranks_per_dp_group * self.data_parallel_size

    def ranks_per_replica(self, capability: str) -> int:
        ep_size = self.expert_parallel_size if capability == "af_ffn" else 1
        return self.tensor_parallel_size * ep_size

    def required_rank_count(self, capability: str) -> int:
        return self.ranks_per_replica(capability) * self.data_parallel_size


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
    after_stage: Literal["a_compute", "expert_compute"]
    participants: tuple[int, ...]
    resource_group: str | None = None
    resource_lane: str | None = None
    network: str | None = None
    payload_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ConfigurationError(
                "domain collective operation must be a non-empty string"
            )
        if not self.participants or any(
            type(rank) is not int or rank < 0 for rank in self.participants
        ):
            raise ConfigurationError(
                f"domain collective {self.operation!r} participants must be "
                "non-empty non-negative ranks"
            )
        if len(self.participants) != len(set(self.participants)):
            raise ConfigurationError(
                f"domain collective {self.operation!r} participants contains duplicates"
            )
        if self.payload_bytes is not None and (
            type(self.payload_bytes) is not int or self.payload_bytes < 0
        ):
            raise ConfigurationError(
                f"domain collective {self.operation!r} payload_bytes must be "
                "a non-negative integer"
            )


@dataclass
class ExecutionDomainConfig:
    name: str
    capability: str
    phases: tuple[str, ...]
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
        """Expose the legacy expert-parallel view used by AF planning."""
        ep = self.parallel.expert_parallel_size
        base = self.parallel.tensor_parallel_size
        world = self.parallel.world_size
        mode = self.parallel.expert_parallel_mode
        if mode is None:
            mode = "rank_reuse" if ep == world and world > 1 else "expert_tp"
        return DomainExpertParallelView(
            mode=mode,
            expert_parallel_size=ep,
            expert_tensor_parallel_size=(1 if mode == "rank_reuse" else base),
            base_world_size=world,
            base_tensor_parallel_size=base,
        )

    def __post_init__(self) -> None:
        for field_name in ("name", "worker_group"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"execution domain field {field_name} must be a non-empty string"
                )
        if self.capability not in WORKER_CAPABILITIES:
            raise ConfigurationError(
                f"execution domain {self.name!r} has unsupported capability "
                f"{self.capability!r}"
            )
        self.phases = _normalize_phases(
            self.phases, f"execution domain {self.name!r} phases"
        )
        instance_names = [instance.name for instance in self.logical_instances]
        if len(instance_names) != len(set(instance_names)):
            raise ConfigurationError(
                f"execution domain {self.name!r} has duplicate logical instance names"
            )


@dataclass
class AFLinkConfig:
    name: str
    phases: tuple[str, ...]
    attention_group: str
    ffn_group: str
    network: str
    resource_group: str
    duplex: str = "full"
    a_to_f_capacity: int | None = None
    f_to_a_capacity: int | None = None
    half_duplex_capacity: int | None = None
    attention_domain: str | None = None
    ffn_domain: str | None = None
    a_to_f_dtype: str | None = None
    f_to_a_dtype: str | None = None

    @field_validator(
        "a_to_f_capacity",
        "f_to_a_capacity",
        "half_duplex_capacity",
        mode="before",
    )
    @classmethod
    def _validate_capacity_type(cls, value: Any) -> int | None:
        if value is not None and type(value) is not int:
            raise ConfigurationError("AF link capacity must be a positive integer")
        return value

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "attention_group",
            "ffn_group",
            "network",
            "resource_group",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"AF link {self.name!r} field {field_name} must be a non-empty string"
                )
        self.phases = _normalize_phases(self.phases, f"AF link {self.name!r} phases")
        if self.duplex not in {"full", "half"}:
            raise ConfigurationError(
                f"AF link {self.name!r} duplex must be 'full' or 'half'"
            )
        capacity_fields = (
            "a_to_f_capacity",
            "f_to_a_capacity",
            "half_duplex_capacity",
        )
        for field_name in capacity_fields:
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ConfigurationError(
                    f"AF link {self.name!r} {field_name} must be a positive integer"
                )
        if self.duplex == "full":
            if self.a_to_f_capacity is None or self.f_to_a_capacity is None:
                raise ConfigurationError(
                    f"AF link {self.name!r} full duplex requires "
                    "a_to_f_capacity and f_to_a_capacity"
                )
            if self.half_duplex_capacity is not None:
                raise ConfigurationError(
                    f"AF link {self.name!r} full duplex cannot set half_duplex_capacity"
                )
        else:
            if self.half_duplex_capacity is None:
                raise ConfigurationError(
                    f"AF link {self.name!r} half duplex requires half_duplex_capacity"
                )
            if self.a_to_f_capacity is not None or self.f_to_a_capacity is not None:
                raise ConfigurationError(
                    f"AF link {self.name!r} half duplex cannot set directional capacities"
                )

    @property
    def effective_capacities(self) -> tuple[int, ...]:
        if self.duplex == "half":
            return (self.half_duplex_capacity,)
        return (self.a_to_f_capacity, self.f_to_a_capacity)

    def communication_resources(self) -> tuple[AFCommunicationResourceConfig, ...]:
        if self.duplex == "half":
            return (
                AFCommunicationResourceConfig(
                    resource_group=self.resource_group,
                    resource_key=(self.resource_group, "shared"),
                    network=self.network,
                    duplex=self.duplex,
                    direction="shared",
                    capacity=self.half_duplex_capacity,
                ),
            )
        return (
            AFCommunicationResourceConfig(
                resource_group=self.resource_group,
                resource_key=(self.resource_group, "a_to_f"),
                network=self.network,
                duplex=self.duplex,
                direction="a_to_f",
                capacity=self.a_to_f_capacity,
            ),
            AFCommunicationResourceConfig(
                resource_group=self.resource_group,
                resource_key=(self.resource_group, "f_to_a"),
                network=self.network,
                duplex=self.duplex,
                direction="f_to_a",
                capacity=self.f_to_a_capacity,
            ),
        )


AF_LATENCY_PROVIDERS = ("injected", "operator_table")
AF_LATENCY_FALLBACKS = ("table_first", "table_only", "analytical_only")


@dataclass
class AFConfig:
    enabled: bool = False
    phases: tuple[str, ...] = AF_PHASE_ORDER
    af_block_size: int = 1
    links: list[AFLinkConfig] = Field(default_factory=list)
    execution_domains: list[ExecutionDomainConfig] = Field(default_factory=list)
    trace_enabled: bool = True
    # ``injected``: the caller supplies compute/transfer providers (tests).
    # ``operator_table``: the engine prices AF stages from the device operator
    # tables in ``data/operator_data`` and AF transfers from the link classes
    # in ``data/topologies/links.yaml``.
    latency_provider: str = "injected"
    # Operator-data backend for AF devices (defaults to the run-wide choice).
    operator_backend: str | None = None
    # ``table_first`` | ``table_only`` | ``analytical_only`` (defaults to the
    # run-wide ``--latency_fallback``).
    latency_fallback: str | None = None
    router_location: str = "a"
    weight_dtype: str | None = None
    activation_dtype: str | None = None
    kv_cache_dtype: str | None = None
    communication_timing: AFCommunicationTimingConfig = Field(
        default_factory=AFCommunicationTimingConfig
    )

    @field_validator("enabled", "trace_enabled", mode="before")
    @classmethod
    def _validate_boolean_type(cls, value: Any, info) -> bool:
        if type(value) is not bool:
            raise ConfigurationError(f"af.{info.field_name} must be a boolean")
        return value

    @field_validator("af_block_size", mode="before")
    @classmethod
    def _validate_block_size_type(cls, value: Any) -> int:
        if type(value) is not int:
            raise ConfigurationError("af_block_size must be a positive integer")
        return value

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ConfigurationError("af.enabled must be a boolean")
        if type(self.trace_enabled) is not bool:
            raise ConfigurationError("af.trace_enabled must be a boolean")
        self.phases = _normalize_phases(self.phases, "af.phases")
        if self.af_block_size < 1:
            raise ConfigurationError("af_block_size must be a positive integer")
        if self.latency_provider not in AF_LATENCY_PROVIDERS:
            raise ConfigurationError(
                "af.latency_provider must be one of " + ", ".join(AF_LATENCY_PROVIDERS)
            )
        if self.operator_backend is not None and (
            not isinstance(self.operator_backend, str) or not self.operator_backend.strip()
        ):
            raise ConfigurationError("af.operator_backend must be a non-empty string")
        if self.latency_fallback is not None and self.latency_fallback not in AF_LATENCY_FALLBACKS:
            raise ConfigurationError(
                "af.latency_fallback must be one of " + ", ".join(AF_LATENCY_FALLBACKS)
            )
        if self.router_location not in {"a", "f"}:
            raise ConfigurationError("af.router_location must be 'a' or 'f'")


def _normalize_phases(phases: tuple[str, ...], context: str) -> tuple[str, ...]:
    values = tuple(phases)
    if not values:
        raise ConfigurationError(
            f"{context} must be a non-empty selection from {sorted(AF_PHASES)}"
        )
    if len(values) != len(set(values)):
        raise ConfigurationError(f"{context} contains duplicate phases")
    unknown = set(values) - AF_PHASES
    if unknown:
        raise ConfigurationError(
            f"{context} contains unsupported phases {sorted(unknown)}; expected values "
            + f"from {sorted(AF_PHASES)}"
        )
    return tuple(phase for phase in AF_PHASE_ORDER if phase in values)


@dataclass
class ClusterConfig:
    num_workers: int
    networks: dict[str, str]
    worker_groups: list[WorkerGroupConfig] = Field(default_factory=list)
    kv_transfer: KVTransferConfig | None = None
    parallel_config: ParallelConfig | None = None
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
        if self.af.enabled:
            self._normalize_execution_domains()
            self._validate_af_topology()
            self._validate_execution_domains()

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

    def _normalize_execution_domains(self) -> None:
        groups = {group.name: group for group in self.worker_groups}
        if self.af.execution_domains:
            for domain in self.af.execution_domains:
                if domain.logical_instances:
                    continue
                group = groups.get(domain.worker_group)
                if (
                    group is None
                    or domain.required_rank_count != 1
                    or group.num_workers != 1
                ):
                    continue
                domain.logical_instances = self._single_rank_instances(group)
            return

        endpoints: dict[tuple[str, str], set[str]] = {}
        for link in self.af.links:
            endpoints.setdefault((link.attention_group, "af_attention"), set()).update(
                link.phases
            )
            endpoints.setdefault((link.ffn_group, "af_ffn"), set()).update(link.phases)
        for group in self.worker_groups:
            for capability, phases in (
                ("prefill", {"prefill"}),
                ("decode_full", {"decode"}),
            ):
                if capability in group.capabilities:
                    endpoints.setdefault((group.name, capability), set()).update(phases)

        domains = []
        for (group_name, capability), phases in sorted(endpoints.items()):
            group = groups.get(group_name)
            if group is None:
                continue
            parallel = DomainParallelConfig(data_parallel_size=group.num_workers)
            domains.append(
                ExecutionDomainConfig(
                    name=f"implicit:{group_name}:{capability}",
                    capability=capability,
                    phases=tuple(phase for phase in AF_PHASE_ORDER if phase in phases),
                    worker_group=group_name,
                    parallel=parallel,
                    logical_instances=self._single_rank_instances(group),
                )
            )
        self.af.execution_domains = domains

    @staticmethod
    def _single_rank_instances(
        group: WorkerGroupConfig,
    ) -> list[LogicalInstanceConfig]:
        return [
            LogicalInstanceConfig(
                name=f"{group.name}:{worker_index}",
                rank_mapping=[
                    DomainRankMapping(
                        domain_rank=worker_index,
                        tp_rank=0,
                        pp_rank=0,
                        dp_rank=worker_index,
                        ep_rank=0,
                        physical_worker=PhysicalWorkerRef(
                            group_name=group.name,
                            worker_index=worker_index,
                        ),
                    )
                ],
            )
            for worker_index in range(group.num_workers)
        ]

    def _validate_execution_domains(self) -> None:
        groups = {group.name: group for group in self.worker_groups}
        domain_names: set[str] = set()
        for domain in self.af.execution_domains:
            if domain.name in domain_names:
                raise ConfigurationError(
                    f"duplicate execution domain name {domain.name!r}"
                )
            domain_names.add(domain.name)
            group = groups.get(domain.worker_group)
            if group is None:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} references unknown worker_group "
                    f"{domain.worker_group!r}"
                )
            if domain.capability not in group.capabilities:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} worker_group "
                    f"{domain.worker_group!r} must declare {domain.capability!r}"
                )
            if not domain.logical_instances:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} requires explicit logical_instances "
                    f"and rank_mapping for required_rank_count="
                    f"{domain.required_rank_count}"
                )
            if domain.parallel.pipeline_parallel_size != 1:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} has "
                    f"pipeline_parallel_size={domain.parallel.pipeline_parallel_size}; "
                    "AF phase one only supports pipeline_parallel_size=1"
                )
            if (
                domain.parallel.expert_parallel_size != 1
                and domain.capability != "af_ffn"
            ):
                raise ConfigurationError(
                    f"execution domain {domain.name!r} expert_parallel_size is only "
                    "valid for af_ffn"
                )
            if len(domain.logical_instances) != domain.parallel.data_parallel_size:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} logical_instances count must "
                    f"equal data_parallel_size={domain.parallel.data_parallel_size}; "
                    f"got {len(domain.logical_instances)}"
                )
            expected_domain_ranks = set(range(domain.required_rank_count))
            seen_dp_ranks: set[int] = set()
            instance_names: set[str] = set()
            physical_owners: dict[str, tuple[str, int]] = {}
            for instance in domain.logical_instances:
                if instance.name in instance_names:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} has duplicate logical instance "
                        f"{instance.name!r}"
                    )
                instance_names.add(instance.name)
                actual_ranks = {
                    mapping.domain_rank for mapping in instance.rank_mapping
                }
                if len(actual_ranks) != len(instance.rank_mapping):
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} logical instance "
                        f"{instance.name!r} has duplicate domain ranks"
                    )
                instance_dp_ranks = {
                    mapping.dp_rank for mapping in instance.rank_mapping
                }
                if len(instance_dp_ranks) != 1:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} logical instance "
                        f"{instance.name!r} must contain exactly one dp_rank; got "
                        f"{sorted(instance_dp_ranks)}"
                    )
                instance_dp_rank = next(iter(instance_dp_ranks))
                if instance_dp_rank in seen_dp_ranks:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} has duplicate dp_rank "
                        f"{instance_dp_rank} across logical instances"
                    )
                seen_dp_ranks.add(instance_dp_rank)
                start = instance_dp_rank * domain.ranks_per_replica
                expected_instance_ranks = set(
                    range(start, start + domain.ranks_per_replica)
                )
                if actual_ranks != expected_instance_ranks:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} logical instance "
                        f"{instance.name!r} rank_mapping must cover dp replica "
                        f"{instance_dp_rank} domain ranks "
                        f"{sorted(expected_instance_ranks)}; got {sorted(actual_ranks)}"
                    )
                for mapping in instance.rank_mapping:
                    self._validate_domain_rank_mapping(domain, group, mapping)
                    physical_identity = mapping.physical_worker.identity
                    previous = physical_owners.get(physical_identity)
                    if previous is not None:
                        previous_instance, previous_rank = previous
                        raise ConfigurationError(
                            f"execution domain {domain.name!r} logical instance "
                            f"{instance.name!r} rank {mapping.domain_rank} maps to physical "
                            f"worker {physical_identity!r}, already mapped by logical instance "
                            f"{previous_instance!r} rank {previous_rank}; placement must be "
                            "strictly one-to-one"
                        )
                    physical_owners[physical_identity] = (
                        instance.name,
                        mapping.domain_rank,
                    )
            expected_dp_ranks = set(range(domain.parallel.data_parallel_size))
            if seen_dp_ranks != expected_dp_ranks:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} logical instances must cover "
                    f"dp ranks {sorted(expected_dp_ranks)}; got {sorted(seen_dp_ranks)}"
                )
            if {
                mapping.domain_rank
                for instance in domain.logical_instances
                for mapping in instance.rank_mapping
            } != expected_domain_ranks:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} rank mappings must cover "
                    f"domain ranks {sorted(expected_domain_ranks)}"
                )
            expected_workers = {
                f"{domain.worker_group}:{worker_index}"
                for worker_index in range(group.num_workers)
            }
            missing_workers = expected_workers - set(physical_owners)
            if missing_workers:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} has unmapped physical workers "
                    f"{sorted(missing_workers)}; every worker in group "
                    f"{domain.worker_group!r} must map to exactly one logical instance rank"
                )
            for collective in domain.communication_plan:
                invalid = set(collective.participants) - expected_domain_ranks
                if invalid:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} collective "
                        f"{collective.operation!r} has out-of-range participants "
                        f"{sorted(invalid)}"
                    )
                if (
                    collective.network is not None
                    and collective.network not in self.networks
                ):
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} collective "
                        f"{collective.operation!r} references unknown network "
                        f"{collective.network!r}"
                    )

    @staticmethod
    def _validate_domain_rank_mapping(
        domain: ExecutionDomainConfig,
        group: WorkerGroupConfig,
        mapping: DomainRankMapping,
    ) -> None:
        parallel = domain.parallel
        ranks_per_replica = domain.ranks_per_replica
        rank_in_replica = mapping.domain_rank % ranks_per_replica
        expected_dp = mapping.domain_rank // ranks_per_replica
        expected_ep = (
            rank_in_replica // parallel.tensor_parallel_size
            if domain.capability == "af_ffn"
            else 0
        )
        expected_tp = rank_in_replica % parallel.tensor_parallel_size
        actual = (
            mapping.tp_rank,
            mapping.pp_rank,
            mapping.dp_rank,
            mapping.ep_rank,
        )
        expected = (expected_tp, 0, expected_dp, expected_ep)
        if actual != expected:
            raise ConfigurationError(
                f"execution domain {domain.name!r} domain_rank={mapping.domain_rank} "
                f"has coordinates {actual}; expected {expected}"
            )
        physical = mapping.physical_worker
        if physical.group_name != domain.worker_group:
            raise ConfigurationError(
                f"execution domain {domain.name!r} rank {mapping.domain_rank} maps to "
                f"worker group {physical.group_name!r}, expected {domain.worker_group!r}"
            )
        if physical.worker_index >= group.num_workers:
            raise ConfigurationError(
                f"execution domain {domain.name!r} rank {mapping.domain_rank} maps to "
                f"out-of-range physical worker {physical.identity}"
            )

    def _validate_af_topology(self) -> None:
        groups = {group.name: group for group in self.worker_groups}
        link_names: set[str] = set()
        phase_links: dict[str, list[AFLinkConfig]] = {
            phase: [] for phase in self.af.phases
        }
        resource_signatures: dict[str, tuple[str, str, tuple[int, ...], str]] = {}

        for link in self.af.links:
            if link.name in link_names:
                raise ConfigurationError(f"duplicate AF link name {link.name!r}")
            link_names.add(link.name)
            unselected = set(link.phases) - set(self.af.phases)
            if unselected:
                raise ConfigurationError(
                    f"AF link {link.name!r} covers unselected phases {sorted(unselected)}"
                )
            for phase in link.phases:
                phase_links[phase].append(link)
            if link.network not in self.networks:
                raise ConfigurationError(
                    f"AF link {link.name!r} references unknown network {link.network!r}"
                )
            if link.attention_group == link.ffn_group:
                raise ConfigurationError(
                    f"AF link {link.name!r} attention_group and ffn_group must differ"
                )
            for endpoint_field, required_capability in (
                ("attention_group", "af_attention"),
                ("ffn_group", "af_ffn"),
            ):
                group_name = getattr(link, endpoint_field)
                group = groups.get(group_name)
                if group is None:
                    raise ConfigurationError(
                        f"AF link {link.name!r} {endpoint_field} references unknown group "
                        + f"{group_name!r}"
                    )
                if required_capability not in group.capabilities:
                    raise ConfigurationError(
                        f"AF link {link.name!r} endpoint group {group_name!r} must declare "
                        + f"{required_capability!r} for phases {list(link.phases)}"
                    )
            for phase in link.phases:
                self._resolve_link_domain(link, phase, "attention")
                self._resolve_link_domain(link, phase, "ffn")
            signature = (
                link.network,
                link.duplex,
                link.effective_capacities,
                link.name,
            )
            existing = resource_signatures.get(link.resource_group)
            if existing is not None and existing[:3] != signature[:3]:
                raise ConfigurationError(
                    f"AF resource_group {link.resource_group!r} conflicts between links "
                    + f"{existing[3]!r} and {link.name!r}"
                )
            resource_signatures[link.resource_group] = signature

        for phase, links in phase_links.items():
            if len(links) != 1:
                raise ConfigurationError(
                    f"AF phase {phase!r} requires exactly one link; found {len(links)}"
                )
        if "prefill" not in self.af.phases and not any(
            "prefill" in group.capabilities for group in self.worker_groups
        ):
            raise ConfigurationError(
                "Decode-only AF requires a worker group with 'prefill' capability"
            )
        if "decode" not in self.af.phases and not any(
            "decode_full" in group.capabilities for group in self.worker_groups
        ):
            raise ConfigurationError(
                "Prefill-only AF requires a worker group with 'decode_full' capability"
            )

    @property
    def phase_to_af_link(self) -> dict[str, AFLinkConfig]:
        if not self.af.enabled:
            return {}
        return {phase: link for link in self.af.links for phase in link.phases}

    @property
    def execution_domains(self) -> dict[str, ExecutionDomainConfig]:
        return {domain.name: domain for domain in self.af.execution_domains}

    @property
    def phase_to_af_domains(
        self,
    ) -> dict[str, tuple[ExecutionDomainConfig, ExecutionDomainConfig]]:
        if not self.af.enabled:
            return {}
        return {
            phase: (
                self._resolve_link_domain(link, phase, "attention"),
                self._resolve_link_domain(link, phase, "ffn"),
            )
            for phase, link in self.phase_to_af_link.items()
        }

    def af_request_owner_identities(self, phase: str) -> tuple[str, ...]:
        if phase not in self.phase_to_af_domains:
            return ()
        attention_domain, _ = self.phase_to_af_domains[phase]
        owners = []
        for instance in attention_domain.logical_instances:
            first_mapping = min(
                instance.rank_mapping, key=lambda mapping: mapping.domain_rank
            )
            owners.append(first_mapping.physical_worker.identity)
        return tuple(dict.fromkeys(owners))

    def _resolve_link_domain(
        self,
        link: AFLinkConfig,
        phase: str,
        side: Literal["attention", "ffn"],
    ) -> ExecutionDomainConfig:
        explicit_name = (
            link.attention_domain if side == "attention" else link.ffn_domain
        )
        capability = "af_attention" if side == "attention" else "af_ffn"
        group_name = link.attention_group if side == "attention" else link.ffn_group
        candidates = [
            domain
            for domain in self.af.execution_domains
            if domain.worker_group == group_name
            and domain.capability == capability
            and phase in domain.phases
            and (explicit_name is None or domain.name == explicit_name)
        ]
        if len(candidates) != 1:
            qualifier = explicit_name or f"{group_name}/{capability}/{phase}"
            raise ConfigurationError(
                f"AF link {link.name!r} {side}_domain {qualifier!r} must resolve "
                f"to exactly one execution domain; found {len(candidates)}"
            )
        return candidates[0]

    @property
    def af_communication_resources(
        self,
    ) -> dict[tuple[str, str], AFCommunicationResourceConfig]:
        resources: dict[tuple[str, str], AFCommunicationResourceConfig] = {}
        if not self.af.enabled:
            return resources
        for link in self.af.links:
            for resource in link.communication_resources():
                resources.setdefault(resource.resource_key, resource)
        return resources

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
