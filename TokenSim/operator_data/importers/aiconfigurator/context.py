"""Shared import options, table assembly and per-source provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from TokenSim.operator_data.package import SourceRecord
from dataclasses import dataclass, field
from functools import cached_property


@dataclass
class ImportContext:
    system_path: Path
    device_id: str
    backend: str
    version: str | None
    nccl_version: str | None
    upstream_commit: str
    gpus_per_node: int | None
    include_custom_allreduce: bool
    kernels: set[str] | None
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sources: dict[str, SourceRecord] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    dropped_dtypes: dict[str, int] = field(default_factory=dict)
    resolved_versions: dict[str, list[str]] = field(default_factory=dict)

    @property
    def system_name(self):
        return self.system_path.name

    @cached_property
    def flat_layout(self):
        return not any(
            (self.system_path / family).is_dir()
            for family in ("gemm", "attention", "moe", "comm")
        )

    def source_for(
        self,
        family: str,
        resolved: Path,
        version_label: str,
        backend_label: str | None = None,
    ) -> str:
        backend = self.backend
        device_id = self.device_id
        flat_layout = self.flat_layout
        sources = self.sources
        system_name = self.system_name
        upstream_commit = self.upstream_commit
        backend_label = backend_label or backend
        if flat_layout:
            source_id = f"aiconfigurator-collector:{device_id}:{family}:{backend_label}:{version_label}"
            reference = str(resolved.resolve())
            notes = (
                "Measured on this device with the AIConfigurator collector "
                f"(commit {upstream_commit}); converted ms -> us."
            )
        else:
            source_id = (
                f"aiconfigurator:{system_name}:{family}:{backend_label}:{version_label}"
            )
            reference = (
                "https://github.com/ai-dynamo/aiconfigurator/tree/"
                f"{upstream_commit}/aic-core/src/aiconfigurator_core/systems/data/{system_name}/{family}/{backend_label}/{version_label}"
            )
            notes = "Measured kernel latencies collected by NVIDIA AIConfigurator (Apache-2.0); converted ms -> us."
        if source_id not in sources:
            sources[source_id] = SourceRecord(
                source_id=source_id,
                grade="A",
                method="measured" if flat_layout else "imported",
                reference=reference,
                notes=notes,
                device=device_id,
                backend=f"{backend_label}:{version_label}",
            )
        return source_id

    def keep_kernel(self, row: Mapping[str, Any]) -> bool:
        kernels = self.kernels
        if kernels is None:
            return True
        kernel = str(row.get("kernel_source", "")).lower()
        return any(token in kernel for token in kernels)
