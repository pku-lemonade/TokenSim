"""Source-derived growth for full-input versus valid MegaMoE work.

Whole-module measurements do not expose phase timers or expert histograms.
This model normalizes physical work to a same-geometry uniform measured
anchor. Expected power-law expert counts, tile traffic and phase overlap are
explicit analytical assumptions, never fitted to serving throughput.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math

from TokenSim.errors import ConfigurationError


def block_m_for_input(rows, ep, experts, topk):
    """Pinned DeepGEMM get_block_config_for_mega_moe, FP8/FP4 path."""
    expected = rows * ep * topk / experts
    covered = expected + math.sqrt(expected)
    block = 16 if expected <= 10 else 32
    if expected > 24:
        blocks = math.ceil(covered / 240)
        if blocks == 1 and covered > 192 and experts / ep < 14:
            blocks = 2
        for candidate in (64, 128, 192, 240):
            block = candidate
            if blocks * candidate >= covered:
                break
    return block


@dataclass(frozen=True)
class MegaMoEWork:
    startup_us: float
    input_us: float
    expert_us: float
    dispatch_us: float
    combine_us: float
    block_ms: tuple[int, ...]
    global_valid_rows: int
    expert_selections: int
    max_expert_tiles: float

    @property
    def latency_us(self):
        # Dispatch and expert work overlap in the fused persistent kernel.
        # Pre-dispatch and the final source-side reduction remain separate.
        return self.startup_us + self.input_us + max(self.expert_us, self.dispatch_us) + self.combine_us


class MegaMoEWorkModel:
    def __init__(self, model, device, ep, *, nvlink_bytes_per_s):
        self.model, self.device, self.ep = model, device, ep
        self.experts = model.moe.num_experts
        self.local_experts = self.experts // ep
        self.topk = model.moe.num_experts_per_tok
        self.hidden, self.inter = model.hidden_size, model.moe_intermediate_size
        self.bandwidth = device.memory_bandwidth_bytes_per_s.value
        self.compute = device.peak_compute_for("mxfp4").value
        self.link = nvlink_bytes_per_s
        if not self.link or self.link <= 0:
            raise ConfigurationError("MegaMoE work growth requires a positive local NVLink bandwidth")
        self.count_cache = OrderedDict()

    def expected_counts(self, global_rows):
        """Expected local spectrum from the collector's inverse-CDF family.

        Midpoint quantiles replace its unrecorded random expert weights. Each
        rank uses the same expected spectrum; the measured anchor retains its
        empirical rank skew. Fractional sub-token occupancy is not rounded up
        to an entire active expert. These are growth weights, not route stats.
        """
        cached = self.count_cache.get(global_rows)
        if cached is not None:
            self.count_cache.move_to_end(global_rows)
            return cached
        if global_rows == 0:
            return (0.0,) * self.local_experts
        alpha = 1.01
        low, high = (1.0, global_rows * 0.8) if global_rows * self.topk > self.experts else (0.01, 2.0)
        weights = [((high ** (1 - alpha) - low ** (1 - alpha)) * ((index + 0.5) / self.local_experts)
                    + low ** (1 - alpha)) ** (1 / (1 - alpha)) for index in range(self.local_experts)]
        total = sum(weights)
        selections = global_rows * self.topk / self.ep
        result = tuple(min(global_rows, selections * weight / total) for weight in weights)
        # The source's per-token unique-top-k bound redistributes overflow.
        remaining = selections - sum(result)
        if remaining > 0:
            available = [index for index, value in enumerate(result) if value < global_rows]
            values = list(result)
            while remaining > 1e-9 and available:
                delta = remaining / len(available)
                next_available = []
                for index in available:
                    addition = min(delta, global_rows - values[index])
                    values[index] += addition
                    remaining -= addition
                    if values[index] < global_rows:
                        next_available.append(index)
                available = next_available
            result = tuple(values)
        if len(self.count_cache) >= 4096:
            self.count_cache.popitem(last=False)
        self.count_cache[global_rows] = result
        return result

    def estimate(self, inputs, valid):
        if len(inputs) != self.ep or len(valid) != self.ep or inputs.keys() != valid.keys():
            raise ConfigurationError("MegaMoE work needs all EP input and valid-row counts")
        if any(n < 0 or not 0 <= valid[rank] <= n for rank, n in inputs.items()):
            raise ConfigurationError("MegaMoE routed rows must fit full inputs")
        global_rows = sum(valid.values())
        counts = self.expected_counts(global_rows)
        blocks = tuple(block_m_for_input(n, self.ep, self.experts, self.topk) for _, n in sorted(inputs.items()))
        expert_us = max_tiles = 0.0
        for block in set(blocks):
            tiles = sum(min(1.0, n) * math.ceil(n / block) for n in counts if n > 0)
            max_tiles = max(max_tiles, tiles)
            # L1 gate/up and L2 down MMA execute complete tiles. Weight/scales
            # are reloaded per M tile; no unmeasured cross-tile L2 credit.
            flops = tiles * block * 6 * self.hidden * self.inter
            weights = tiles * 3 * self.hidden * self.inter * (0.5 + 1 / 32)
            activations = sum(counts) * (self.hidden * (1 + 1 / 32 + 2) + self.inter * (2 + 2 / 32))
            expert_us = max(expert_us, max(flops / self.compute, (weights + activations) / self.bandwidth) * 1e6)
        full_max, valid_max = max(inputs.values()), max(valid.values())
        # BF16 X -> FP8 X/scales, routing staging and two top-k scans.
        input_bytes = full_max * (self.hidden * (2 + 1 + 1 / 32) + self.topk * (12 + 16))
        remote_fraction = (self.ep - 1) / self.ep
        dispatch_bytes = valid_max * self.topk * self.hidden * (1 + 1 / 32) * remote_fraction
        combine_bytes = valid_max * self.topk * self.hidden * 2 * remote_fraction
        local_combine_bytes = full_max * (self.topk * 8 + self.hidden * 2) + valid_max * self.topk * self.hidden * 2
        return MegaMoEWork(
            3 * self.device.analytical["kernel_launch_us"], input_bytes / self.bandwidth * 1e6,
            expert_us, dispatch_bytes / self.link * 1e6,
            (combine_bytes / self.link + local_combine_bytes / self.bandwidth) * 1e6,
            blocks, global_rows, global_rows * self.topk, max_tiles,
        )

    def describe(self):
        return {
            "basis": "pinned DeepGEMM input-selected BLOCK_M, valid expert tasks, pre-dispatch/scan and source-side combine",
            "normalization": "target physical phase work / matching full-input uniform phase work; whole measured module anchor retained",
            "expert_spectrum": "inverse-CDF power_law_1.01 midpoint quantiles; expected local spectrum, empirical source skew retained in anchor",
            "grade": "D source-derived composition",
            "fitted_parameters": False,
            "uncertainty": "no measured component timers/histograms, historical DeepGEMM revision, tile cache traffic or unequal-rank phase overlap",
            "cache_credit": "no cross-M-tile weight L2 reuse credited",
            "full_input_work": "input preparation/scans, block choice and output reduction preserved even for padding",
        }
