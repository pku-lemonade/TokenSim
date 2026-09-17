"""DeviceSpec — the full specification of a single accelerator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import positive_number, require
from TokenSim.hardware.device.dtypes import compute_dtype, normalize_dtype
from TokenSim.hardware.device.sourced_value import SourcedValue

DEFAULT_ANALYTICAL_PARAMETERS: dict[str, dict[str, float]] = {
    # Fraction of peak throughput a well-tuned kernel reaches at large shapes,
    # bandwidth efficiency for streaming kernels, and a per-kernel launch cost.
    # These are project defaults (grade D) until a calibration replaces them.
    "nvidia_gpu": {
        "gemm_mfu": 0.70,
        "attention_mfu": 0.45,
        "moe_mfu": 0.50,
        "memory_efficiency": 0.85,
        "elementwise_memory_efficiency": 0.70,
        "kernel_launch_us": 4.0,
        "gemm_small_m_knee": 64.0,
    },
    "groq_tsp": {
        "gemm_mfu": 0.80,
        "attention_mfu": 0.60,
        "moe_mfu": 0.70,
        "memory_efficiency": 0.90,
        "elementwise_memory_efficiency": 0.90,
        "kernel_launch_us": 0.2,
        "gemm_small_m_knee": 1.0,
    },
    "generic": {
        "gemm_mfu": 0.60,
        "attention_mfu": 0.40,
        "moe_mfu": 0.40,
        "memory_efficiency": 0.80,
        "elementwise_memory_efficiency": 0.60,
        "kernel_launch_us": 5.0,
        "gemm_small_m_knee": 64.0,
    },
}

SUPPORTED_FAMILIES = tuple(DEFAULT_ANALYTICAL_PARAMETERS)


@dataclass(frozen=True)
class DeviceSpec:
    device_id: str
    family: str
    display_name: str
    aliases: tuple[str, ...]
    peak_compute: Mapping[str, SourcedValue]
    memory_kind: str
    memory_capacity_bytes: SourcedValue
    memory_bandwidth_bytes_per_s: SourcedValue
    scale_up_link: str | None
    scale_up_ports: int
    host_link: str | None
    step_overhead_us: SourcedValue
    analytical: Mapping[str, float]
    on_chip_memory_bytes: SourcedValue | None = None
    on_chip_bandwidth_bytes_per_s: SourcedValue | None = None
    sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    tdp_w: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], context: str = "device") -> "DeviceSpec":
        device_id = str(require(raw, "device_id", context))
        context = f"device {device_id!r}"
        if raw.get("schema_version", 1) != 1:
            raise ConfigurationError(f"{context}: unsupported schema_version")
        family = str(raw.get("family", "generic"))
        if family not in SUPPORTED_FAMILIES:
            raise ConfigurationError(
                f"{context}: family must be one of {SUPPORTED_FAMILIES}, got {family!r}"
            )
        peak_raw = require(raw, "peak_compute", context)
        if not isinstance(peak_raw, Mapping) or not peak_raw:
            raise ConfigurationError(f"{context}: peak_compute must be a non-empty mapping")
        peak = {
            normalize_dtype(dtype): SourcedValue.parse(value, f"peak_compute.{dtype}", context)
            for dtype, value in peak_raw.items()
        }
        memory = require(raw, "memory", context)
        if not isinstance(memory, Mapping):
            raise ConfigurationError(f"{context}: memory must be a mapping")
        on_chip = raw.get("on_chip_memory") or {}
        interconnect = raw.get("interconnect") or {}
        analytical = dict(DEFAULT_ANALYTICAL_PARAMETERS[family])
        for key, value in (raw.get("analytical") or {}).items():
            analytical[str(key)] = float(value)
        sources = {str(k): dict(v) for k, v in (raw.get("sources") or {}).items()}
        step_overhead = SourcedValue.parse_optional_zero(
            raw.get("step_overhead_us"),
            "step_overhead_us",
            context,
            default_source="assumption:no-step-overhead",
        )
        power = raw.get("power") or {}
        aliases = tuple(str(alias) for alias in raw.get("aliases", []))
        return cls(
            device_id=device_id,
            family=family,
            display_name=str(raw.get("display_name", device_id)),
            aliases=aliases,
            peak_compute=peak,
            memory_kind=str(memory.get("kind", "unknown")),
            memory_capacity_bytes=SourcedValue.parse(
                require(memory, "capacity_bytes", context), "memory.capacity_bytes", context
            ),
            memory_bandwidth_bytes_per_s=SourcedValue.parse(
                require(memory, "bandwidth_bytes_per_s", context),
                "memory.bandwidth_bytes_per_s",
                context,
            ),
            scale_up_link=(
                str(interconnect["scale_up_link"]) if interconnect.get("scale_up_link") else None
            ),
            scale_up_ports=int(interconnect.get("scale_up_ports", 1)),
            host_link=str(interconnect["host_link"]) if interconnect.get("host_link") else None,
            step_overhead_us=step_overhead,
            analytical=analytical,
            on_chip_memory_bytes=(
                SourcedValue.parse(on_chip["capacity_bytes"], "on_chip_memory.capacity_bytes", context)
                if on_chip.get("capacity_bytes")
                else None
            ),
            on_chip_bandwidth_bytes_per_s=(
                SourcedValue.parse(
                    on_chip["bandwidth_bytes_per_s"],
                    "on_chip_memory.bandwidth_bytes_per_s",
                    context,
                )
                if on_chip.get("bandwidth_bytes_per_s")
                else None
            ),
            sources=sources,
            tdp_w=float(power["tdp_w"]) if power.get("tdp_w") is not None else None,
            extra={
                str(k): v
                for k, v in raw.items()
                if k
                not in {
                    "schema_version",
                    "device_id",
                    "family",
                    "display_name",
                    "aliases",
                    "peak_compute",
                    "memory",
                    "on_chip_memory",
                    "interconnect",
                    "analytical",
                    "sources",
                    "step_overhead_us",
                    "power",
                }
            },
        )

    @classmethod
    def simple(
        cls,
        device_id: str,
        *,
        family: str = "generic",
        peak_flops: float = 1e15,
        memory_capacity_bytes: float = 80 * (1 << 30),
        memory_bandwidth_bytes_per_s: float = 2e12,
        scale_up_link: str | None = None,
        host_link: str | None = None,
        step_overhead_us: float = 0.0,
        dtypes: Iterable[str] = ("fp16", "bf16"),
        aliases: Iterable[str] = (),
        analytical: Mapping[str, float] | None = None,
        source_id: str = "test-fixture",
    ) -> "DeviceSpec":
        """Build an in-memory device for tests and quick experiments."""
        params = dict(DEFAULT_ANALYTICAL_PARAMETERS.get(family, DEFAULT_ANALYTICAL_PARAMETERS["generic"]))
        params.update(analytical or {})
        return cls(
            device_id=device_id,
            family=family,
            display_name=device_id,
            aliases=tuple(aliases),
            peak_compute={
                normalize_dtype(dtype): SourcedValue(peak_flops, source_id, "D") for dtype in dtypes
            },
            memory_kind="test",
            memory_capacity_bytes=SourcedValue(memory_capacity_bytes, source_id, "D"),
            memory_bandwidth_bytes_per_s=SourcedValue(memory_bandwidth_bytes_per_s, source_id, "D"),
            scale_up_link=scale_up_link,
            scale_up_ports=1,
            host_link=host_link,
            step_overhead_us=SourcedValue(step_overhead_us, source_id, "D"),
            analytical=params,
        )

    # -- queries -----------------------------------------------------------

    def peak_compute_for(self, dtype: str) -> SourcedValue:
        """Peak dense throughput (ops/s) of the pipe a storage dtype runs on."""
        pipe = compute_dtype(dtype)
        if pipe in self.peak_compute:
            return self.peak_compute[pipe]
        # bf16 and fp16 share tensor pipes on every device we model.
        fallback = {"bf16": "fp16", "fp16": "bf16", "tf32": "fp32"}.get(pipe)
        if fallback and fallback in self.peak_compute:
            return self.peak_compute[fallback]
        raise ConfigurationError(
            f"device {self.device_id!r} has no peak compute for dtype {dtype!r} (pipe {pipe!r})"
        )

    @property
    def weights_resident_on_chip(self) -> bool:
        """True for SRAM-resident architectures (Groq TSP)."""
        return self.family == "groq_tsp"

    @property
    def names(self) -> tuple[str, ...]:
        return (self.device_id, self.display_name, *self.aliases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "family": self.family,
            "display_name": self.display_name,
            "peak_compute": {k: v.value for k, v in self.peak_compute.items()},
            "memory_capacity_bytes": self.memory_capacity_bytes.value,
            "memory_bandwidth_bytes_per_s": self.memory_bandwidth_bytes_per_s.value,
            "scale_up_link": self.scale_up_link,
            "host_link": self.host_link,
            "step_overhead_us": self.step_overhead_us.value,
        }
