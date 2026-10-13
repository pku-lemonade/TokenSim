"""Explicit analytical cost for compressed attention's hyper-residual output reduction.

The target applies this operation before selecting logits rows. The MTP
predictor defers it to compute_logits. No paired hc_head timing table exists;
keep the FP32 gating work and BF16 traffic visible rather than omitting it.
"""

from TokenSim.errors import ConfigurationError
from TokenSim.latency.base import roofline_latency_us


class ResidualOutputEstimator:
    def __init__(self, backend, *, scope):
        if backend.model.components is None or scope not in ("target", "draft"):
            raise ConfigurationError(
                "hc_head requires compressed attention and a target/draft scope"
            )
        self.backend, self.scope = backend, scope
        self.hidden = backend.model.hidden_size
        self.hc = backend.model.components.hc_mult
        self.weight_bytes = 4 * (self.hc * self.hc * self.hidden + self.hc + 1)
        self.reset_stats()

    def reset_stats(self):
        self.calls = self.rows = 0
        self.seconds = 0.0

    def estimate(self, rows):
        rows = int(rows)
        if rows <= 0:
            return 0.0
        device = self.backend.device
        # Preserve the existing dedicated draft estimate: FP32 projection,
        # row RMS/gating/reweight, one BF16 residual read and dense output.
        bytes_ = self.weight_bytes + rows * self.hidden * (2 * self.hc + 2)
        flops = rows * self.hidden * (2 * self.hc * self.hc + 8 * self.hc)
        latency_us = roofline_latency_us(device, bytes_, flops)
        self.backend.stats.record(
            "target_hc_head" if self.scope == "target" else "mtp_hc_head", "analytical"
        )
        self.calls += 1
        self.rows += rows
        self.seconds += latency_us * 1e-6
        return latency_us

    def describe(self):
        return {
            "scope": self.scope,
            "source_grade": "D",
            "source": "analytical FP32 gating and BF16 hyper-residual reduction; no matched module measurement",
            "hidden_size": self.hidden,
            "hc_mult": self.hc,
            "parameter_bytes": self.weight_bytes,
            "input_output_bytes_per_row": self.hidden * (2 * self.hc + 2),
            "flops_per_row": self.hidden * (2 * self.hc * self.hc + 8 * self.hc),
            "cost": "max(math_seconds, memory_seconds) + existing kernel_launch_us",
            "excludes": "final/shared RMSNorm and selected-row lm_head, billed separately",
        }
