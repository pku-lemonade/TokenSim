"""Source-derived growth for full-input versus valid MegaMoE work.

Whole-module measurements do not expose phase timers or expert histograms.
This model normalizes physical work to a same-geometry uniform measured
anchor. Expected power-law expert counts, tile traffic and phase overlap are
explicit analytical assumptions, never fitted to serving throughput.
"""

from dataclasses import dataclass
import math

import numpy as np

from TokenSim.errors import ConfigurationError
from TokenSim.latency.deepseek_v4_moe_spectrum import BLOCK_MS, SEEDS, integer_spectra, spectrum_geometry


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
    expert_mma_rows: float = 0.0
    expert_flops: float = 0.0
    issued_weight_bytes: float = 0.0
    fused_expert_dispatch_us: float | None = None

    @property
    def latency_us(self):
        # Dispatch and expert work overlap in the fused persistent kernel.
        # Pre-dispatch and the final source-side reduction remain separate.
        fused = self.fused_expert_dispatch_us
        return self.startup_us + self.input_us + (max(self.expert_us, self.dispatch_us) if fused is None else fused) + self.combine_us


class MegaMoEWorkModel:
    def __init__(self, model, device, ep, *, nvlink_bytes_per_s):
        self.model, self.device, self.ep = model, device, ep
        self.experts = model.moe.num_experts
        self.local_experts = self.experts // ep
        self.topk = model.moe.num_experts_per_tok
        self.hidden, self.inter = model.hidden_size, model.moe_intermediate_size
        self.bandwidth = device.memory_bandwidth_bytes_per_s.value
        # The routed kernel uses MXF8F6F4 with FP8 activations, not the
        # pure FP4 MXF4/NVF4 opcode. Use the catalog FP8 theoretical pipe;
        # mixed-input issue efficiency remains an unmeasured approximation.
        self.compute = device.peak_compute_for("fp8_block").value
        self.link = nvlink_bytes_per_s
        self._terms = {}
        if not self.link or self.link <= 0:
            raise ConfigurationError("MegaMoE work growth requires a positive local NVLink bandwidth")

    def integer_spectra(self, global_rows):
        return integer_spectra(global_rows, self.experts, self.topk, self.ep)

    def geometry(self, global_rows):
        return spectrum_geometry(global_rows, self.experts, self.topk, self.ep)

    def _spectrum_terms(self, global_rows):
        """Per-sample, per-rank work terms of one global spectrum, as lists."""
        terms = self._terms.get(global_rows)
        if terms is None:
            geometry = self.geometry(global_rows)
            activation_bytes = geometry.selections * (
                self.hidden * (1 + 1 / 32 + 2) + self.inter * (2 + 2 / 32)
            )
            math_us = geometry.mma_rows * (6 * self.hidden * self.inter / self.compute * 1e6)
            # Pinned TaskInfo aligns valid_m to16 and the kernel updates its
            # swapped MMA N descriptor per task. Every candidate BLOCK_M is a
            # multiple of16, so total MMA rows are independent of M partitioning.
            terms = (
                float(geometry.mma_rows.max(axis=1).mean()),
                geometry.selections,
                math_us.tolist(),
                activation_bytes.tolist(),
                geometry.m_tiles.tolist(),
            )
            if len(self._terms) > 65536:
                self._terms.clear()
            self._terms[global_rows] = terms
        return terms

    def estimate(self, inputs, valid):
        if len(inputs) != self.ep or len(valid) != self.ep or inputs.keys() != valid.keys():
            raise ConfigurationError("MegaMoE work needs all EP input and valid-row counts")
        if any(n < 0 or not 0 <= valid[rank] <= n for rank, n in inputs.items()):
            raise ConfigurationError("MegaMoE routed rows must fit full inputs")
        global_rows = sum(valid.values())
        mma_rows, selections, math_us, activation_bytes, m_tiles = self._spectrum_terms(global_rows)
        blocks = tuple(block_m_for_input(n, self.ep, self.experts, self.topk) for _, n in sorted(inputs.items()))
        flops = mma_rows * 6 * self.hidden * self.inter
        weights_per_tile = 3 * self.hidden * self.inter * (0.5 + 1 / 32)
        block_indices = [BLOCK_MS.index(block) for block in blocks]
        bandwidth = self.bandwidth
        # Unknown serving hot-rank identity must not inherit the collector's
        # deliberate swap to rank0. Every placement retains its slowest rank;
        # only those critical work costs are averaged, never full input rows.
        # With one BLOCK_M for every input rank all placements see the same
        # rank set, so their critical costs repeat.
        rotations = 1 if len(set(block_indices)) == 1 else self.ep
        critical, critical_tiles = [], []
        for rotation in range(rotations):
            for sample_math, sample_bytes, sample_tiles in zip(math_us, activation_bytes, m_tiles):
                slowest, most_tiles = None, None
                for position, block in enumerate(block_indices):
                    rank = (position + rotation) % self.ep
                    tiles = sample_tiles[rank][block]
                    memory = (tiles * weights_per_tile + sample_bytes[rank]) / bandwidth * 1e6
                    work = sample_math[rank] if sample_math[rank] >= memory else memory
                    if slowest is None or work > slowest:
                        slowest = work
                    if most_tiles is None or tiles > most_tiles:
                        most_tiles = tiles
                critical.append(slowest)
                critical_tiles.append(most_tiles)
        if rotations == 1:
            critical *= self.ep
            critical_tiles *= self.ep
        # NumPy means keep the original pairwise summation order.
        expert_us = float(np.mean(critical))
        max_tiles = float(np.mean(critical_tiles))
        weight_bytes = max_tiles * weights_per_tile
        full_max, valid_max = max(inputs.values()), max(valid.values())
        # BF16 X -> FP8 X/scales, routing staging and two top-k scans.
        input_bytes = full_max * (self.hidden * (2 + 1 + 1 / 32) + self.topk * (12 + 16))
        remote_fraction = (self.ep - 1) / self.ep
        dispatch_bytes = valid_max * self.topk * self.hidden * (1 + 1 / 32) * remote_fraction
        combine_bytes = valid_max * self.topk * self.hidden * 2 * remote_fraction
        local_combine_bytes = full_max * (self.topk * 8 + self.hidden * 2) + valid_max * self.topk * self.hidden * 2
        dispatch_us = dispatch_bytes / self.link * 1e6
        return MegaMoEWork(
            3 * self.device.analytical["kernel_launch_us"], input_bytes / self.bandwidth * 1e6,
            expert_us, dispatch_us,
            (combine_bytes / self.link + local_combine_bytes / self.bandwidth) * 1e6,
            blocks, global_rows, global_rows * self.topk, max_tiles,
            mma_rows, flops, weight_bytes,
            float(np.mean([value if value >= dispatch_us else dispatch_us for value in critical])),
        )

    def describe(self):
        return {
            "basis": "pinned DeepGEMM input-selected BLOCK_M, valid expert tasks, pre-dispatch/scan and source-side combine",
            "normalization": "target physical phase work / matching full-input uniform phase work; whole measured module anchor retained",
            "expert_spectrum": "global power_law_1.01 integer spectra; fixed MT19937 float32 draws0/1/2, rounding/clipping/round-robin conservation; synthetic, not recovered historical routes",
            "spectrum_samples": list(SEEDS),
            "spectrum_placement": "mean critical costs over all cyclic hot-rank placements; each retains max-rank work and full-input BLOCK_M; no mean-row module substitution",
            "spectrum_rounding": "float32 inverse-CDF; NumPy/Torch reduction differences may move boundary counts, exact integer selection bounds retained",
            "mma_rows": "valid expert task tails aligned to16; per-expert16*ceil(received/16); mean critical-rank arithmetic over synthetic samples",
            "mma_opcode": "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale; FP8 activation/FP4 weight, UMMA_K32",
            "compute_pipe": "catalog FP8 theoretical peak for the mixed MXF8F6F4 path; not pure FP4 storage-dtype peak",
            "compute_flops_per_s": self.compute,
            "compute_rate_grade": "D pipe-family inference from pinned opcode/operands; mixed-input issue cycles and efficiency unmeasured",
            "weight_traffic": "full weight/scales per M task; dynamic MMA tails do not reduce TMA loads; issued bytes are not measured HBM misses",
            "grade": "D source-derived composition",
            "fitted_parameters": False,
            "uncertainty": "no measured component timers/histograms, historical DeepGEMM revision, tile cache traffic or unequal-rank phase overlap",
            "cache_credit": "no cross-M-tile weight L2 reuse credited",
            "full_input_work": "input preparation/scans, block choice and output reduction preserved even for padding",
        }
