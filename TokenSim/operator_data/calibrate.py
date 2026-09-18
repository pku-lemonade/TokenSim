"""Fit analytical efficiency parameters against measured operator tables.

The analytical models expose a handful of interpretable knobs per device
family (``gemm_mfu``, ``attention_mfu``, ``moe_mfu``, ``memory_efficiency``,
``elementwise_memory_efficiency``, ``kernel_launch_us``,
``gemm_small_m_knee``). Calibration searches those knobs to minimise the mean
absolute log error between the analytical estimate and measured rows, holding
out a deterministic subset of rows to report generalisation error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from TokenSim.hardware.device import DeviceSpec
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import TABLE_SPECS

TABLE_PARAMETERS: Mapping[str, tuple[str, ...]] = {
    "gemm": ("gemm_mfu", "memory_efficiency", "kernel_launch_us", "gemm_small_m_knee"),
    "context_attention": ("attention_mfu", "memory_efficiency", "kernel_launch_us"),
    "generation_attention": ("attention_mfu", "memory_efficiency", "kernel_launch_us"),
    "moe": ("moe_mfu", "memory_efficiency", "kernel_launch_us"),
    "elementwise": ("elementwise_memory_efficiency", "kernel_launch_us"),
}

PARAMETER_GRIDS: Mapping[str, tuple[float, ...]] = {
    "gemm_mfu": tuple(x / 100 for x in range(20, 101, 5)),
    "attention_mfu": tuple(x / 100 for x in range(5, 101, 5)),
    "moe_mfu": tuple(x / 100 for x in range(10, 101, 5)),
    "memory_efficiency": tuple(x / 100 for x in range(30, 101, 5)),
    "elementwise_memory_efficiency": tuple(x / 100 for x in range(20, 101, 5)),
    "kernel_launch_us": (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0),
    "gemm_small_m_knee": (1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0),
}


@dataclass
class ErrorStats:
    count: int
    mean_abs_log_error: float | None
    mean_relative_error: float | None
    median_relative_error: float | None
    p90_relative_error: float | None
    max_relative_error: float | None
    bias: float | None
    worst: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_pairs(cls, pairs: Sequence[tuple[Mapping[str, Any], float, float]]) -> "ErrorStats":
        if not pairs:
            return cls(0, None, None, None, None, None, None)
        rel = [(pred - meas) / meas for _, meas, pred in pairs if meas > 0]
        abs_rel = sorted(abs(r) for r in rel)
        logs = [abs(math.log(pred / meas)) for _, meas, pred in pairs if meas > 0 and pred > 0]
        worst = sorted(
            ({"key": dict(k), "measured_us": m, "predicted_us": p, "relative_error": (p - m) / m} for k, m, p in pairs if m > 0),
            key=lambda item: abs(item["relative_error"]),
            reverse=True,
        )[:10]
        return cls(
            count=len(pairs),
            mean_abs_log_error=mean(logs) if logs else None,
            mean_relative_error=mean(abs_rel) if abs_rel else None,
            median_relative_error=median(abs_rel) if abs_rel else None,
            p90_relative_error=abs_rel[min(len(abs_rel) - 1, int(0.9 * len(abs_rel)))] if abs_rel else None,
            max_relative_error=abs_rel[-1] if abs_rel else None,
            bias=mean(rel) if rel else None,
            worst=worst,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_abs_log_error": self.mean_abs_log_error,
            "mean_relative_error": self.mean_relative_error,
            "median_relative_error": self.median_relative_error,
            "p90_relative_error": self.p90_relative_error,
            "max_relative_error": self.max_relative_error,
            "bias": self.bias,
            "worst": self.worst,
        }


@dataclass
class TableCalibration:
    table: str
    parameters: dict[str, float]
    fit: ErrorStats
    holdout: ErrorStats
    baseline_fit: ErrorStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "parameters": self.parameters,
            "fit": self.fit.to_dict(),
            "holdout": self.holdout.to_dict(),
            "baseline_fit": {k: v for k, v in self.baseline_fit.to_dict().items() if k != "worst"},
        }


def _split(rows: Sequence[Mapping[str, Any]], holdout_fraction: float, seed: int) -> tuple[list, list]:
    fit: list = []
    holdout: list = []
    for index, row in enumerate(rows):
        bucket = (hash((seed, index)) % 1000) / 1000.0
        (holdout if bucket < holdout_fraction else fit).append(row)
    return fit, holdout


def _evaluate(device: DeviceSpec, table: str, rows: Iterable[Mapping[str, Any]], gated: bool) -> list[tuple[Mapping[str, Any], float, float]]:
    model = analytical_model_for(device.family)
    spec = TABLE_SPECS[table]
    pairs = []
    for row in rows:
        key = {f: row[f] for f in spec.key_fields}
        try:
            estimate = model.estimate(table, key, device, Calibration(), gated=gated)
        except Exception:
            continue
        pairs.append((key, float(row["latency_us"]), estimate.latency_us))
    return pairs


def _objective(pairs: Sequence[tuple[Mapping[str, Any], float, float]]) -> float:
    logs = [abs(math.log(pred / meas)) for _, meas, pred in pairs if meas > 0 and pred > 0]
    return mean(logs) if logs else float("inf")


def calibrate_table(
    device: DeviceSpec,
    package: OperatorDataPackage,
    table: str,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    max_rows: int = 4000,
    gated: bool = True,
    rounds: int = 3,
) -> TableCalibration | None:
    rows = list(package.tables.get(table, ()))
    if not rows or table not in TABLE_PARAMETERS:
        return None
    if len(rows) > max_rows:
        step = len(rows) / max_rows
        rows = [rows[int(i * step)] for i in range(max_rows)]
    fit_rows, holdout_rows = _split(rows, holdout_fraction, seed)
    if not fit_rows:
        return None
    baseline = ErrorStats.from_pairs(_evaluate(device, table, fit_rows, gated))
    params = dict(device.analytical)
    parameter_names = TABLE_PARAMETERS[table]
    current = replace(device, analytical=params)
    best_score = _objective(_evaluate(current, table, fit_rows, gated))
    for _ in range(rounds):
        improved = False
        for name in parameter_names:
            for value in PARAMETER_GRIDS[name]:
                trial_params = {**params, name: value}
                trial = replace(device, analytical=trial_params)
                score = _objective(_evaluate(trial, table, fit_rows, gated))
                if score < best_score - 1e-9:
                    best_score, params, improved = score, trial_params, True
        if not improved:
            break
    tuned = replace(device, analytical=params)
    return TableCalibration(
        table=table,
        parameters={name: params[name] for name in parameter_names},
        fit=ErrorStats.from_pairs(_evaluate(tuned, table, fit_rows, gated)),
        holdout=ErrorStats.from_pairs(_evaluate(tuned, table, holdout_rows, gated)),
        baseline_fit=baseline,
    )


def calibrate_package(device: DeviceSpec, package: OperatorDataPackage, **kwargs: Any) -> dict[str, TableCalibration]:
    results: dict[str, TableCalibration] = {}
    for table in TABLE_PARAMETERS:
        result = calibrate_table(device, package, table, **kwargs)
        if result is not None:
            results[table] = result
    return results
