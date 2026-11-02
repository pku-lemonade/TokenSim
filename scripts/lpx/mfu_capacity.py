"""Exact resident-capacity screening over contiguous pipeline partitions."""

import numpy as np

from TokenSim.config.validation import integer
from TokenSim.errors import ConfigurationError


class ResidentCapacity:
    """Reuse native memory vectors without rebuilding runtimes for every count.

    Feasibility includes native block rounding and the compiler KV watermark.
    Every possible stage count is checked; splitting a stage can change block
    rounding, so feasibility is not inferred from the minimum stage count.
    """

    def __init__(self, footprint):
        self.footprint = footprint
        self.layers = len(footprint.weights)
        pairs = [
            (start, end) for end in range(1, self.layers + 1) for start in range(end)
        ]
        self.starts = np.array([start for start, _ in pairs])
        self.ends = np.array([end for _, end in pairs])
        try:
            for name in ("weights", "kv", "blocks"):
                values = np.asarray(getattr(footprint, name), dtype=np.int64)
                prefix = np.vstack(
                    [np.zeros((1, values.shape[1]), dtype=np.int64), values.cumsum(0)]
                )
                setattr(self, name, prefix[self.ends] - prefix[self.starts])
        except (OverflowError, TypeError, ValueError) as exc:
            raise ConfigurationError("capacity vectors must fit signed int64") from exc
        boundary = (self.starts == 0).astype(int) + (self.ends == self.layers).astype(
            int
        )
        if footprint.tied_embeddings:
            boundary = np.minimum(boundary, 1)
        self.fixed = self.weights + np.asarray(footprint.buffers)
        self.fixed += boundary[:, None] * np.asarray(footprint.embedding)[None, :]
        self.ends_at = [
            np.flatnonzero(self.ends == end) for end in range(self.layers + 1)
        ]

    def segment_loads(self, resident):
        integer(resident, "resident request count", minimum=1)
        maximum = int(self.kv.max(initial=0))
        if maximum and resident > np.iinfo(np.int64).max // maximum:
            raise ConfigurationError("resident KV arithmetic exceeds signed int64")
        required = np.zeros_like(self.blocks)
        mask = self.blocks > 0
        required[mask] = (
            self.kv[mask] * resident + self.blocks[mask] - 1
        ) // self.blocks[mask]
        watermark = self.footprint.kv_watermark
        slots = required.copy()
        nonzero = required > 0
        if watermark:
            slots[nonzero] = np.maximum(
                required[nonzero],
                ((required[nonzero] - 1) / (1 - watermark)).astype(np.int64) + 1,
            )
            while np.any(slots - (watermark * slots).astype(np.int64) < required):
                slots += slots - (watermark * slots).astype(np.int64) < required
            removable = lambda: (slots > 0) & (
                slots - 1 - (watermark * (slots - 1)).astype(np.int64) >= required
            )
            while np.any(remove := removable()):
                slots -= remove
        rounded = np.where(mask, slots * self.blocks, self.kv * resident)
        return self.fixed + rounded

    def feasible_stages(self, resident, maximum=None):
        maximum = self.layers if maximum is None else maximum
        integer(maximum, "maximum pipeline stages", minimum=1)
        maximum = min(maximum, self.layers)
        valid = np.all(
            self.segment_loads(resident) <= np.asarray(self.footprint.capacities),
            axis=1,
        )
        reachable = [0] * (self.layers + 1)
        reachable[0] = 1
        keep = (1 << (maximum + 1)) - 1
        for end in range(1, self.layers + 1):
            for edge in self.ends_at[end]:
                if valid[edge]:
                    reachable[end] |= reachable[int(self.starts[edge])] << 1
            reachable[end] &= keep
        return tuple(
            stage for stage in range(1, maximum + 1) if reachable[-1] & (1 << stage)
        )

    def frontier(self, maximum_resident, maximum_stages=None):
        """Record the largest admitted count, retaining an explicit search limit."""
        integer(maximum_resident, "capacity search request limit", minimum=1)
        result = {}
        last_feasible = ()
        for resident in range(1, maximum_resident + 1):
            last_feasible = self.feasible_stages(resident, maximum_stages)
            if not last_feasible:
                break
            for stages in last_feasible:
                result[stages] = resident
        return dict(
            maximum_resident_by_pp=result,
            searched_through_resident=resident,
            request_limit_reached=bool(last_feasible),
        )
