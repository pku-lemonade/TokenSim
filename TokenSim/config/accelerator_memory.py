from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from TokenSim.errors import ConfigurationError
from TokenSim.mooncake.config import (
    SUPPORTED_CONNECTORS,
    MooncakeConfig,
    parse_mooncake_config,
)


@dataclass(frozen=True)
class AcceleratorMemoryProfile:
    base_hardware: str
    accelerator_memory_preset: str
    memory_type: str
    capacity_gib_per_card: float
    bandwidth_tb_s_per_card: float


@dataclass(frozen=True)
class EffectiveHardwareProfile:
    base_hardware: str
    accelerator_memory_preset: str
    memory_type: str
    capacity_gib_per_card: float
    bandwidth_tb_s_per_card: float
    worker_count: int
    shared_offload_capacity_gib: float
    offload_tier: str

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


_PRESETS = {
    ("H100", "HBF1"): ("GDDR7", 32.0, 1.792),
    ("H100", "HBF1-48GiB"): ("GDDR7", 48.0, 1.344),
    ("H100", "HBF2"): ("HBM3e", 48.0, 1.5),
    ("H200", "HBF1"): ("GDDR7", 32.0, 1.792),
    ("H200", "HBF2"): ("HBM3e", 48.0, 1.5),
    ("B200", "HBF1"): ("GDDR7", 64.0, 3.584),
    ("B200", "HBF1-48GiB"): ("GDDR7", 96.0, 2.688),
    ("B200", "HBF2"): ("HBM3e", 96.0, 4.0),
}


def resolve_accelerator_memory_profile(
    base_hardware: str,
    accelerator_memory_preset: str,
    *,
    target_hardware: str | None = None,
) -> AcceleratorMemoryProfile:
    if target_hardware is not None and target_hardware != base_hardware:
        raise ConfigurationError(
            "accelerator memory preset target conflicts with cluster hardware: "
            + f"target_hardware={target_hardware!r}, base_hardware={base_hardware!r}"
        )
    values = _PRESETS.get((base_hardware, accelerator_memory_preset))
    if values is None:
        raise ConfigurationError(
            "unsupported accelerator memory preset target: "
            + f"base_hardware={base_hardware!r}, "
            + f"accelerator_memory_preset={accelerator_memory_preset!r}"
        )
    memory_type, capacity_gib, bandwidth_tb_s = values
    return AcceleratorMemoryProfile(
        base_hardware=base_hardware,
        accelerator_memory_preset=accelerator_memory_preset,
        memory_type=memory_type,
        capacity_gib_per_card=capacity_gib,
        bandwidth_tb_s_per_card=bandwidth_tb_s,
    )


def resolve_run_hardware_profile(
    cluster: Any,
    kv_transfer_config: Any,
) -> EffectiveHardwareProfile | None:
    contexts = [
        config
        for config in _iter_mooncake_configs(
            kv_transfer_config.kv_connector,
            kv_transfer_config.kv_connector_extra_config,
        )
        if config.accelerator_memory_preset is not None
    ]
    if not contexts:
        return None

    signatures = {
        (
            config.accelerator_memory_preset,
            config.target_hardware,
            config.offload_tier,
            _shared_offload_capacity_gib(config),
        )
        for config in contexts
    }
    if len(signatures) != 1:
        raise ConfigurationError(
            "Mooncake connectors must use one consistent accelerator memory preset "
            "and shared offload profile per run"
        )
    preset, target_hardware, offload_tier, shared_capacity_gib = signatures.pop()

    base_hardwares = {group.hardware for group in cluster.worker_groups}
    if len(base_hardwares) != 1:
        raise ConfigurationError(
            "accelerator memory presets require one base hardware across all worker groups; "
            + f"got {sorted(base_hardwares)}"
        )
    base_hardware = base_hardwares.pop()
    profile = resolve_accelerator_memory_profile(
        base_hardware,
        preset,
        target_hardware=target_hardware,
    )
    return EffectiveHardwareProfile(
        **asdict(profile),
        worker_count=int(cluster.num_workers),
        shared_offload_capacity_gib=shared_capacity_gib,
        offload_tier=offload_tier,
    )


def apply_accelerator_memory_profile(
    roofline: Any,
    profile: EffectiveHardwareProfile | AcceleratorMemoryProfile,
) -> Any:
    try:
        base_hardware = roofline.hardwares[profile.base_hardware]
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown roofline hardware {profile.base_hardware!r}"
        ) from exc

    effective_hardware = copy(base_hardware)
    bandwidth = profile.bandwidth_tb_s_per_card
    effective_hardware.Capacity = profile.capacity_gib_per_card
    effective_hardware.BW_TBs = bandwidth
    effective_hardware.MM_BW_TBs = bandwidth
    effective_hardware.MM_GP_BW_TBs = bandwidth
    effective_hardware.MV_BW_TBs = bandwidth
    effective_hardware.MM_Sweet_Point = effective_hardware.MM_TFLOPS / bandwidth
    effective_hardware.MM_GP_Sweet_Point = effective_hardware.MM_GP_TFLOPS / bandwidth
    effective_hardware.MV_Sweet_Point = effective_hardware.MV_TFLOPS / bandwidth
    effective_hardware.Memory_Type = profile.memory_type

    run_hardwares = dict(roofline.hardwares)
    run_hardwares[profile.base_hardware] = effective_hardware
    roofline.hardwares = run_hardwares
    return effective_hardware


def configure_run_hardware(
    roofline: Any,
    cluster: Any,
    kv_transfer_config: Any,
) -> EffectiveHardwareProfile | None:
    profile = resolve_run_hardware_profile(cluster, kv_transfer_config)
    if profile is not None:
        apply_accelerator_memory_profile(roofline, profile)
    return profile


def _iter_mooncake_configs(
    connector_name: str | None,
    extra_config: dict[str, Any] | None,
) -> Iterator[MooncakeConfig]:
    extra_config = dict(extra_config or {})
    if connector_name == "MultiConnector":
        for child in extra_config.get("connectors", []):
            yield from _iter_mooncake_configs(
                child.get("kv_connector"),
                child.get("kv_connector_extra_config", {}),
            )
        return
    if connector_name in SUPPORTED_CONNECTORS:
        yield parse_mooncake_config(extra_config)


def _shared_offload_capacity_gib(config: MooncakeConfig) -> float:
    if config.offload_tier == "hbf":
        return float(config.hbf_capacity_gb)
    return float(config.ssd_capacity_gb)
