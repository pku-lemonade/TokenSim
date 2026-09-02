import json
from pathlib import Path
from typing import Any

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


@dataclass
class WorkerGroupConfig:
    role: str | None = None
    hardware: str = ""
    num_workers: int = 0
    network: str = "net1"
    name: str | None = None
    capabilities: tuple[str, ...] = ()

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
            )
            for _ in range(self.num_workers)
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


@dataclass
class AFConfig:
    enabled: bool = False
    phases: tuple[str, ...] = AF_PHASE_ORDER
    af_block_size: int = 1
    links: list[AFLinkConfig] = Field(default_factory=list)
    trace_enabled: bool = True
    latency_provider: str = "injected"
    operator_data_path: str | None = None
    operator_manifest_path: str | None = None
    operator_data_backend: str = "analytical"
    operator_link_group: str = "af-fabric"
    operator_group_size: int = 2

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

    @field_validator("operator_group_size", mode="before")
    @classmethod
    def _validate_operator_group_size_type(cls, value: Any) -> int:
        if type(value) is not int:
            raise ConfigurationError(
                "af.operator_group_size must be a positive integer"
            )
        return value

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ConfigurationError("af.enabled must be a boolean")
        if type(self.trace_enabled) is not bool:
            raise ConfigurationError("af.trace_enabled must be a boolean")
        self.phases = _normalize_phases(self.phases, "af.phases")
        if self.af_block_size < 1:
            raise ConfigurationError("af_block_size must be a positive integer")
        if self.latency_provider not in {"injected", "operator_data"}:
            raise ConfigurationError(
                "af.latency_provider must be 'injected' or 'operator_data'"
            )
        if self.operator_group_size < 1:
            raise ConfigurationError(
                "af.operator_group_size must be a positive integer"
            )
        for field_name in (
            "operator_data_backend",
            "operator_link_group",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"af.{field_name} must be a non-empty string")
        if self.latency_provider == "operator_data":
            missing = [
                field_name
                for field_name in ("operator_data_path", "operator_manifest_path")
                if not isinstance(getattr(self, field_name), str)
                or not getattr(self, field_name).strip()
            ]
            if missing:
                raise ConfigurationError(
                    "af.latency_provider='operator_data' requires " + ", ".join(missing)
                )


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
            self._validate_af_topology()

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
