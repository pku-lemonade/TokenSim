"""Assemble a measured operator package from an AIConfigurator system or collector run."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.package import OperatorDataPackage, PackageMeta
from .context import ImportContext
from .discovery import _system_gpus_per_node
from .gemm import import_gemm
from .attention import import_attention
from .moe import import_moe
from .communication import import_expert_all_to_all
from .communication import import_collectives


def import_aiconfigurator_system(
    system_dir: str | Path,
    *,
    device_id: str,
    backend: str = "trtllm",
    version: str | None = None,
    nccl_version: str | None = None,
    dataset_version: str | None = None,
    include_custom_allreduce: bool = True,
    kernel_filter: Iterable[str] | None = None,
    upstream_commit: str = "unknown",
    gpus_per_node: int | None = None,
    module_sources: Mapping[str, str | Path] | None = None,
) -> OperatorDataPackage:
    """Build an :class:`OperatorDataPackage` from one AIConfigurator system directory.

    ``gpus_per_node`` is only used for SGLang's wide-EP DeepEP tables, which
    record ``node_num`` but not ``ep_size``; the EP group is assumed to span all
    GPUs of the participating nodes. It defaults to the upstream system
    description's ``num_gpus_per_node``.
    """
    system_path = Path(system_dir)
    if not system_path.is_dir():
        raise ConfigurationError(
            f"AIConfigurator system directory not found: {system_path}"
        )
    if gpus_per_node is None:
        gpus_per_node = _system_gpus_per_node(system_path)
    context = ImportContext(
        system_path=system_path,
        device_id=device_id,
        backend=backend,
        version=version,
        nccl_version=nccl_version,
        upstream_commit=upstream_commit,
        gpus_per_node=gpus_per_node,
        include_custom_allreduce=include_custom_allreduce,
        kernels={k.lower() for k in kernel_filter} if kernel_filter else None,
    )
    import_gemm(context)
    import_attention(context)
    import_moe(context)
    import_expert_all_to_all(context)
    import_collectives(context)
    dropped = context.dropped
    dropped_dtypes = context.dropped_dtypes
    flat_layout = context.flat_layout
    resolved_versions = context.resolved_versions
    sources = context.sources
    system_name = context.system_name
    tables = context.tables

    from ..modules import import_modules

    import_modules(
        system_path,
        backend,
        version,
        upstream_commit,
        tables,
        sources,
        target_device=device_id,
        overrides=module_sources,
    )
    if not tables:
        raise ConfigurationError(
            f"no AIConfigurator tables found under {system_path} for backend={backend!r} version={version!r}"
        )
    meta = PackageMeta(
        dataset_version=dataset_version
        or f"aiconfigurator-{system_name}-{backend}-{version or 'latest'}",
        device_id=device_id,
        backend=backend,
        framework_version=version or (resolved_versions.get("gemm") or [""])[0],
        sources=sources,
        notes=(
            (
                "Collected on this device with the AIConfigurator collector. "
                if flat_layout
                else "Imported from NVIDIA AIConfigurator measured kernel database. "
            )
            + f"Resolved versions: {resolved_versions}. Rows dropped (unsupported dtype/kernel): {dropped}."
        ),
        extra={
            "aiconfigurator_system": system_name,
            "aiconfigurator_layout": "collector_run" if flat_layout else "database",
            "aiconfigurator_commit": upstream_commit,
            "resolved_versions": resolved_versions,
            "dropped_rows": dropped,
            "dropped_dtypes": dropped_dtypes,
            "license": "Apache-2.0 (NVIDIA AIConfigurator); see data/operator_data/LICENSES.md",
        },
    )
    return OperatorDataPackage.from_rows(meta, tables)
