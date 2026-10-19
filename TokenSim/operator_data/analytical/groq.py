"""Deterministic streaming model for the Groq TSP (LPU).

Weights, activations and KV cache all live in on-chip SRAM, so the memory term
uses the SRAM bandwidth and there is no HBM weight-streaming floor. The model
also reports whether the rank-local weights fit in SRAM; the generator turns
that into an explicit ``fits_on_chip`` flag in the analysis table instead of
silently producing a latency for an impossible placement.
"""

from __future__ import annotations
from TokenSim.dtypes import CAST_OPERATIONS

from typing import Any, Mapping

from TokenSim.hardware.device import DeviceSpec
from TokenSim.operator_data.analytical.base import (
    AnalyticalEstimate,
    Calibration,
    register_model,
    roofline_estimate,
)
from TokenSim.operator_data.workload import work_for_table


class GroqTSPModel:
    family = "groq_tsp"

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
        sram = device.on_chip_memory_bytes or device.memory_capacity_bytes
        bandwidth = (
            device.on_chip_bandwidth_bytes_per_s or device.memory_bandwidth_bytes_per_s
        ).value
        source_id = (
            f"analytical:{self.family}:{device.device_id}:{calibration.calibration_id}"
        )
        overhead_us = float(params.get("kernel_launch_us", 0.2))
        gated = bool(context.get("gated", True))
        imbalance = float(context.get("imbalance", 1.0))

        if table in {"gemm", "grouped_gemm"}:
            dtype = str(key["dtype"])
            work = work_for_table(table, key)
            efficiency = float(params.get("gemm_mfu", 0.8))
            peak = device.peak_compute_for(dtype).value
        elif table in {
            "context_attention",
            "generation_attention",
            "ragged_generation_attention",
        }:
            dtype = str(key["attn_dtype"])
            work = work_for_table(table, key)
            efficiency = float(params.get("attention_mfu", 0.6))
            peak = device.peak_compute_for(dtype).value
        elif table in {"moe", "routed_experts"}:
            dtype = str(key["dtype"])
            work = work_for_table(table, key, gated=gated, imbalance=imbalance)
            efficiency = float(params.get("moe_mfu", 0.7))
            peak = device.peak_compute_for(dtype).value
        elif table == "elementwise":
            dtype = str(key["dtype"])
            work = work_for_table(table, key)
            efficiency = 0.8
            peak = device.peak_compute_for(
                "fp32" if key["op_name"] in CAST_OPERATIONS else dtype
            ).value
        else:
            raise ValueError(
                f"{self.family} analytical model does not cover table {table!r}"
            )

        fits = work.weight_bytes <= sram.value
        effective_flops = work.flops
        vector_time = 0.0
        if work.details.get("dequant_elements") and "nvfp4" not in device.peak_compute:
            vector_ops = float(work.details["dequant_elements"]) * float(
                params.get("packed_dequant_ops_per_element", 2)
            )
            vector_peak = device.peak_compute_for("fp32").value * float(
                params.get("dequant_efficiency", 0.8)
            )
            vector_time = vector_ops / vector_peak
            effective_flops = max(effective_flops, vector_time * peak * efficiency)
        return roofline_estimate(
            work=work,
            peak_ops_per_s=peak,
            bandwidth_bytes_per_s=bandwidth,
            compute_efficiency=efficiency,
            memory_efficiency=float(params.get("memory_efficiency", 0.9)),
            overhead_us=overhead_us,
            calibration=calibration,
            source_id=source_id,
            compute_flops_override=effective_flops,
            details={
                "fits_on_chip": fits,
                "sram_bytes": sram.value,
                "weight_bytes": work.weight_bytes,
                "packed_dequant_time_us": vector_time * 1e6,
            },
        )


register_model(GroqTSPModel())
