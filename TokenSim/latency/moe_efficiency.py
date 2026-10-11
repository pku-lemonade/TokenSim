"""Measured uniform shape efficiency, transferred only to expert/dispatch work.

Whole-module data does not measure component efficiency. The allocation is
grade D: complete source anchors and all full-input/common work stay intact.
No valid-rank mean is queried as a replacement module shape.
"""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
import math

from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class PhaseTransfer:
    latency_us: float
    common_us: float
    expert_us: float
    efficiency_ratio: float
    match: str
    capacity: int | None


def fused_expert_work(work):
    value = work.fused_expert_dispatch_us
    return max(work.expert_us, work.dispatch_us) if value is None else value


class ExpertEfficiencyTransfer:
    def __init__(self, work_model, rows):
        self.work_model = work_model
        self.phase_rows = defaultdict(list)
        groups = defaultdict(list)
        for row in rows:
            capacity = row.get("effective_num_max_tokens_per_rank")
            # Synthetic/test or legacy rows without capacity metadata are not
            # treated as same-geometry source efficiency measurements.
            if not isinstance(capacity, int) or capacity < row["num_tokens"]:
                continue
            n = row["num_tokens"]
            uniform = {rank: n for rank in range(work_model.ep)}
            work = work_model.estimate(uniform, uniform)
            expert = fused_expert_work(work)
            gamma = row["latency_us"] / work.latency_us
            if expert <= 0 or not math.isfinite(gamma) or gamma <= 0:
                raise ConfigurationError("invalid measured MegaMoE efficiency source")
            self.phase_rows[row["phase"]].append((n, capacity))
            groups[row["phase"], capacity].append((expert, gamma, n))
        self.curves = {}
        for key, values in groups.items():
            if len(values) < 3:
                continue
            values.sort()
            # Different source shapes at the same work coordinate cannot
            # identify a unique efficiency; retain the original normalization.
            if len({value[0] for value in values}) != len(values):
                continue
            self.curves[key] = tuple(values)
        for values in self.phase_rows.values():
            values.sort()

    def capacity_for(self, phase, rows):
        samples = self.phase_rows.get(phase, ())
        if not samples:
            return None
        counts = [value[0] for value in samples]
        index = bisect_left(counts, rows)
        if index < len(samples) and counts[index] == rows:
            capacity = samples[index][1]
        elif index == 0:
            capacity = samples[0][1]
        elif index == len(samples):
            capacity = samples[-1][1]
        elif samples[index - 1][1] == samples[index][1]:
            capacity = samples[index][1]
        else:
            return None  # Explicitly do not blend different buffer capacities.
        return capacity if rows <= capacity else None

    @staticmethod
    def gamma(curve, expert):
        coordinate = [sample[0] for sample in curve]
        index = bisect_left(coordinate, expert)
        if index < len(curve) and coordinate[index] == expert:
            return curve[index][1], "exact_efficiency"
        if index == 0:
            return curve[0][1], "boundary_efficiency"
        if index == len(curve):
            return curve[-1][1], "boundary_efficiency"
        a, b = curve[index - 1], curve[index]
        fraction = (expert - a[0]) / (b[0] - a[0])
        return (1 - fraction) * a[1] + fraction * b[1], "interpolated_efficiency"

    def price(self, anchor, source, target, *, anchor_rows, phase):
        if not math.isfinite(anchor) or anchor < 0 or source.latency_us <= 0:
            raise ConfigurationError("invalid native MegaMoE phase-transfer anchor")
        factor = anchor / source.latency_us
        common = factor * (target.startup_us + target.input_us + target.combine_us)
        target_expert, source_expert = fused_expert_work(target), fused_expert_work(
            source
        )
        capacity = self.capacity_for(phase, anchor_rows)
        curve = self.curves.get((phase, capacity))
        ratio, match = 1.0, "unchanged_no_same_capacity_curve"
        if target_expert == 0:
            match = "zero_routed_expert_work"
        elif curve and source_expert > 0:
            source_gamma, _ = self.gamma(curve, source_expert)
            target_gamma, match = self.gamma(curve, target_expert)
            ratio = target_gamma / source_gamma
        expert = factor * target_expert * ratio
        total = common + expert
        if not all(
            math.isfinite(value) and value >= 0
            for value in (common, expert, total, ratio)
        ):
            raise ConfigurationError("invalid MegaMoE transferred component cost")
        return PhaseTransfer(total, common, expert, ratio, match, capacity)

    def describe(self):
        return {
            "grade": "D: whole uniform module efficiency allocated to expert/fused-dispatch phase; component efficiency not measured",
            "source": "same phase/effective-capacity raw module rows; at least 3 distinct expert-work coordinates",
            "common_scope": "startup, full-input preparation/scans and final combine retain source-anchor efficiency",
            "expert_scope": "critical expert/fused-dispatch work after integer sampling and full-input BLOCK_M; normalized measured shape efficiency ratio",
            "identity": "uniform source anchor retained exactly; source-relative efficiency ratio equals 1 at identical expert work",
            "fallback": "missing/singleton capacity, ambiguous work coordinate, cross-capacity or cross-buffer interpolation retain original physical normalization",
            "capacity_curves": [
                {"phase": phase, "effective_capacity": capacity, "points": len(curve)}
                for (phase, capacity), curve in sorted(self.curves.items())
            ],
            "serving_fit": False,
            "cache_credit": False,
            "mean_rank_module_query": False,
        }
