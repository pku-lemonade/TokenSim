"""Deterministic synthetic integer spectra for native MegaMoE growth.

The collector's original timing histograms are unrecorded. These fixed draws
model its distribution family and integer constraints, not serving routes.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from TokenSim.errors import ConfigurationError


SEEDS = (0, 1, 2)
BLOCK_MS = (16, 32, 64, 128, 192, 240)


def _adjust(counts, remaining, step, upper):
    while remaining:
        progressed = False
        for row in counts:
            eligible = (
                [i for i, n in enumerate(row) if n < upper]
                if step > 0
                else [i for i, n in enumerate(row) if n > 0]
            )
            if not eligible:
                continue
            index = (
                min(eligible, key=lambda i: row[i])
                if step > 0
                else max(eligible, key=lambda i: row[i])
            )
            row[index] += step
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise ConfigurationError(
                "synthetic expert counts cannot conserve unique top-k"
            )


@lru_cache(maxsize=4096)
def integer_spectra(global_rows, experts, topk, ep):
    if (
        global_rows < 0
        or experts < 1
        or ep < 1
        or experts % ep
        or not 1 <= topk <= experts
    ):
        raise ConfigurationError("invalid native integer expert spectrum geometry")
    if not global_rows:
        result = np.zeros((len(SEEDS), ep, experts // ep), dtype=np.int64)
    else:
        samples = []
        low, high = (
            (1, global_rows * 0.8) if global_rows * topk > experts else (0.01, 2)
        )
        exponent = 1 - 1.01
        for seed in SEEDS:
            # Match the source CPU float32 inverse-CDF family with a local
            # MT19937 stream. Float reduction may differ across libraries;
            # exact conservation is restored after rounding and clipping.
            integers = np.random.RandomState(seed).randint(
                0, 2**32, size=experts, dtype=np.uint32
            )
            uniforms = (integers & np.uint32(2**24 - 1)).astype(
                np.float32
            ) / np.float32(2**24)
            weights = np.power(
                np.float32(high**exponent - low**exponent) * uniforms
                + np.float32(low**exponent),
                np.float32(1 / exponent),
            )
            values = np.rint(
                weights / weights.sum(dtype=np.float32) * np.float32(global_rows * topk)
            ).astype(np.int64)
            overflow = int(np.maximum(values - global_rows, 0).sum())
            counts = np.minimum(values, global_rows).reshape(ep, experts // ep).tolist()
            if overflow:
                _adjust(counts, overflow, 1, global_rows)
            difference = global_rows * topk - sum(sum(row) for row in counts)
            if difference:
                _adjust(
                    counts, abs(difference), 1 if difference > 0 else -1, global_rows
                )
            heaviest = max(range(ep), key=lambda rank: sum(counts[rank]))
            counts[0], counts[heaviest] = counts[heaviest], counts[0]
            samples.append(counts)
        result = np.asarray(samples, dtype=np.int64)
    if not np.all(result.sum(axis=(1, 2)) == global_rows * topk) or not np.all(
        (result >= 0) & (result <= global_rows)
    ):
        raise ConfigurationError("synthetic expert spectrum violates selection bounds")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SpectrumGeometry:
    mma_rows: np.ndarray
    selections: np.ndarray
    m_tiles: np.ndarray


@lru_cache(maxsize=4096)
def spectrum_geometry(global_rows, experts, topk, ep):
    counts = integer_spectra(global_rows, experts, topk, ep)
    mma_rows = (((counts + 15) // 16) * 16).sum(axis=2)
    selections = counts.sum(axis=2)
    m_tiles = np.stack(
        [((counts + block - 1) // block).sum(axis=2) for block in BLOCK_MS], axis=2
    )
    for array in (mma_rows, selections, m_tiles):
        array.setflags(write=False)
    return SpectrumGeometry(mma_rows, selections, m_tiles)
