from __future__ import annotations

from TokenSim.config.model_config import ModelSpec
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceSpec
from TokenSim.latency.base import LatencyBackend, backend_prefill_len
from TokenSim.latency.operator_table import (
    FALLBACK_POLICIES,
    OperatorStats,
    OperatorTableLatencyBackend,
)
from TokenSim.moe.placement import ExpertPlacement
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.parallel import ParallelCommunicator

BACKEND_TYPES = ("operator_table", "analytical")


def build_latency_backend(
    backend_type: str,
    *,
    device: DeviceSpec,
    model: ModelSpec,
    package: OperatorDataPackage | None = None,
    parallel_config: ParallelConfig | None = None,
    rank_info: ParallelRankInfo | None = None,
    communicator: ParallelCommunicator | None = None,
    expert_placement: ExpertPlacement | None = None,
    fallback: str = "table_first",
    decode_context_bucket: int = 128,
    random_seed: int = 0,
) -> LatencyBackend:
    """Create the latency backend for one worker.

    ``operator_table`` composes per-operator tables with analytical fallback
    (``fallback`` selects ``table_first``, ``table_only`` or ``analytical_only``);
    ``analytical`` is shorthand for ``operator_table`` with ``analytical_only``.
    """
    if backend_type == "analytical":
        backend_type, fallback = "operator_table", "analytical_only"
    if backend_type not in BACKEND_TYPES:
        raise ConfigurationError(
            f"unsupported latency backend {backend_type!r}; expected one of {BACKEND_TYPES}"
        )
    if fallback not in FALLBACK_POLICIES:
        raise ConfigurationError(f"unsupported fallback policy {fallback!r}; expected one of {FALLBACK_POLICIES}")
    return OperatorTableLatencyBackend(
        device=device,
        model=model,
        parallel_config=parallel_config,
        rank_info=rank_info,
        communicator=communicator,
        expert_placement=expert_placement,
        package=package,
        fallback=fallback,
        decode_context_bucket=decode_context_bucket,
        random_seed=random_seed,
    )


__all__ = [
    "BACKEND_TYPES",
    "FALLBACK_POLICIES",
    "LatencyBackend",
    "OperatorStats",
    "OperatorTableLatencyBackend",
    "backend_prefill_len",
    "build_latency_backend",
]
