"""Worker."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass

from TokenSim.config.constants import (
    LEGACY_ROLE_CAPABILITIES,
    WORKER_CAPABILITIES,
    WORKER_CAPABILITY_ORDER,
    WORKER_ROLES,
    WORKER_ROLE_PREFIXES,
)
from TokenSim.config.execution_domain import PhysicalWorkerRef
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


@dataclass(config=ConfigDict(extra="forbid"))
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


@dataclass(config=ConfigDict(extra="forbid"))
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
        if (
            self.device_indices is not None
            and len(self.device_indices) != self.num_workers
        ):
            raise ConfigurationError(
                f"{context} device_indices must have num_workers entries"
            )
        if self.operator_backend is not None and (
            not isinstance(self.operator_backend, str)
            or not self.operator_backend.strip()
        ):
            raise ConfigurationError(
                f"{context} operator_backend must be a non-empty string"
            )

        for index in self.device_indices or ():
            integer(index, "device_index")

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
