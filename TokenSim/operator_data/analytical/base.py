from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceSpec
from TokenSim.operator_data.workload import OperatorWork


@dataclass(frozen=True)
class Calibration:
    """Affine correction of a roofline estimate: ``latency = m * roofline + b``."""

    multiplier: float = 1.0
    offset_us: float = 0.0
    calibration_id: str = "uncalibrated"
    grade: str = "D"

    def apply(self, roofline_us: float) -> float:
        value = roofline_us * self.multiplier + self.offset_us
        if value < 0:
            raise ConfigurationError(
                f"calibration {self.calibration_id!r} produced a negative latency"
            )
        return value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "Calibration":
        if not raw:
            return cls()
        return cls(
            multiplier=float(raw.get("multiplier", 1.0)),
            offset_us=float(raw.get("offset_us", 0.0)),
            calibration_id=str(raw.get("calibration_id", "uncalibrated")),
            grade=str(raw.get("grade", "D")).upper(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "multiplier": self.multiplier,
            "offset_us": self.offset_us,
            "calibration_id": self.calibration_id,
            "grade": self.grade,
        }


@dataclass(frozen=True)
class AnalyticalEstimate:
    latency_us: float
    compute_time_us: float
    memory_time_us: float
    overhead_us: float
    bottleneck: str
    work: OperatorWork
    calibration_id: str
    source_id: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def roofline_us(self) -> float:
        return max(self.compute_time_us, self.memory_time_us) + self.overhead_us

    def analysis_row(self) -> dict[str, Any]:
        return {
            "flops": self.work.flops,
            "weight_bytes": self.work.weight_bytes,
            "activation_bytes": self.work.activation_bytes,
            "kv_bytes": self.work.kv_bytes,
            "compute_time_us": self.compute_time_us,
            "memory_time_us": self.memory_time_us,
            "overhead_us": self.overhead_us,
            "roofline_us": self.roofline_us,
            "latency_us": self.latency_us,
            "bottleneck": self.bottleneck,
            "calibration_id": self.calibration_id,
        }


class AnalyticalModel(Protocol):
    family: str

    def estimate(
        self,
        table: str,
        key: Mapping[str, Any],
        device: DeviceSpec,
        calibration: Calibration | None = None,
        **context: Any,
    ) -> AnalyticalEstimate: ...


def roofline_estimate(
    *,
    work: OperatorWork,
    peak_ops_per_s: float,
    bandwidth_bytes_per_s: float,
    compute_efficiency: float,
    memory_efficiency: float,
    overhead_us: float,
    calibration: Calibration,
    source_id: str,
    details: Mapping[str, Any] | None = None,
    compute_flops_override: float | None = None,
) -> AnalyticalEstimate:
    if peak_ops_per_s <= 0 or bandwidth_bytes_per_s <= 0:
        raise ConfigurationError("peak compute and bandwidth must be positive")
    flops = work.flops if compute_flops_override is None else compute_flops_override
    compute_time_us = flops / (peak_ops_per_s * max(compute_efficiency, 1e-9)) * 1e6
    memory_time_us = work.total_bytes / (bandwidth_bytes_per_s * max(memory_efficiency, 1e-9)) * 1e6
    bottleneck = "compute" if compute_time_us >= memory_time_us else "memory"
    roofline_us = max(compute_time_us, memory_time_us) + overhead_us
    return AnalyticalEstimate(
        latency_us=calibration.apply(roofline_us),
        compute_time_us=compute_time_us,
        memory_time_us=memory_time_us,
        overhead_us=overhead_us,
        bottleneck=bottleneck,
        work=work,
        calibration_id=calibration.calibration_id,
        source_id=source_id,
        details=dict(details or {}),
    )


_MODELS: dict[str, AnalyticalModel] = {}


def register_model(model: AnalyticalModel) -> None:
    _MODELS[model.family] = model


def analytical_model_for(family: str) -> AnalyticalModel:
    # Import lazily so that the family modules can import this one.
    if not _MODELS:
        from TokenSim.operator_data.analytical import gpu, groq  # noqa: F401

    try:
        return _MODELS[family]
    except KeyError:
        return _MODELS["generic"]
