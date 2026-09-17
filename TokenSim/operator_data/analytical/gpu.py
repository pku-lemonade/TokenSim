"""Roofline-with-efficiency model for HBM/GDDR GPUs (NVIDIA, generic accelerators).

Peak throughput and bandwidth come from the device spec; the efficiency
parameters are project defaults (grade D) that ``operator_data.cli calibrate``
replaces with fitted values per device and table.
"""

from __future__ import annotations

from typing import Any, Mapping

from TokenSim.hardware.device import DeviceSpec
from TokenSim.operator_data.analytical.base import (
    AnalyticalEstimate,
    Calibration,
    register_model,
    roofline_estimate,
)
from TokenSim.operator_data.workload import gemm_work, work_for_table


class GPURooflineModel:
    family = "nvidia_gpu"

    def estimate(
        self,
        table: str,
        key: Mapping[str, Any],
        device: DeviceSpec,
        calibration: Calibration | None = None,
        **context: Any,
    ) -> AnalyticalEstimate:
        calibration = calibration or Calibration()
        params = device.analytical
        bandwidth = device.memory_bandwidth_bytes_per_s.value
        source_id = f"analytical:{self.family}:{device.device_id}:{calibration.calibration_id}"
        launch_us = float(params.get("kernel_launch_us", 4.0))
        gated = bool(context.get("gated", True))
        imbalance = float(context.get("imbalance", 1.0))

        if table == "gemm":
            m, n, k = int(key["m"]), int(key["n"]), int(key["k"])
            dtype = str(key["dtype"])
            work = gemm_work(m, n, k, dtype)
            knee = float(params.get("gemm_small_m_knee", 64.0))
            # Tensor-core tiles process at least ``knee`` rows per pass.
            effective_flops = 2.0 * max(float(m), knee) * n * k
            return roofline_estimate(
                work=work,
                peak_ops_per_s=device.peak_compute_for(dtype).value,
                bandwidth_bytes_per_s=bandwidth,
                compute_efficiency=float(params.get("gemm_mfu", 0.7)),
                memory_efficiency=float(params.get("memory_efficiency", 0.85)),
                overhead_us=launch_us,
                calibration=calibration,
                source_id=source_id,
                details={"effective_m": max(m, knee)},
                compute_flops_override=effective_flops,
            )
        if table in {"context_attention", "generation_attention"}:
            work = work_for_table(table, key)
            dtype = str(key["attn_dtype"])
            efficiency = float(params.get("attention_mfu", 0.45))
            return roofline_estimate(
                work=work,
                peak_ops_per_s=device.peak_compute_for(dtype).value,
                bandwidth_bytes_per_s=bandwidth,
                compute_efficiency=efficiency,
                memory_efficiency=float(params.get("memory_efficiency", 0.85)),
                overhead_us=launch_us,
                calibration=calibration,
                source_id=source_id,
            )
        if table == "moe":
            work = work_for_table(table, key, gated=gated, imbalance=imbalance)
            dtype = str(key["dtype"])
            return roofline_estimate(
                work=work,
                peak_ops_per_s=device.peak_compute_for(dtype).value,
                bandwidth_bytes_per_s=bandwidth,
                compute_efficiency=float(params.get("moe_mfu", 0.5)),
                memory_efficiency=float(params.get("memory_efficiency", 0.85)),
                # grouped GEMM issues gate/up and down kernels plus permutation
                overhead_us=3 * launch_us,
                calibration=calibration,
                source_id=source_id,
            )
        if table == "elementwise":
            work = work_for_table(table, key)
            dtype = str(key["dtype"])
            return roofline_estimate(
                work=work,
                peak_ops_per_s=device.peak_compute_for("fp32").value
                if "fp32" in device.peak_compute
                else device.peak_compute_for(dtype).value,
                bandwidth_bytes_per_s=bandwidth,
                compute_efficiency=0.5,
                memory_efficiency=float(params.get("elementwise_memory_efficiency", 0.7)),
                overhead_us=launch_us,
                calibration=calibration,
                source_id=source_id,
            )
        raise ValueError(f"{self.family} analytical model does not cover table {table!r}")


class GenericRooflineModel(GPURooflineModel):
    family = "generic"


register_model(GPURooflineModel())
register_model(GenericRooflineModel())
