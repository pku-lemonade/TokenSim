"""Portable offline candidate for the collector's integer count construction."""

import numpy as np


def adjust(counts, remaining, step, upper):
    while remaining:
        progressed = False
        for row in counts:
            eligible = [i for i, n in enumerate(row) if n < upper] if step > 0 else [i for i, n in enumerate(row) if n > 0]
            if not eligible:
                continue
            index = min(eligible, key=lambda i: row[i]) if step > 0 else max(eligible, key=lambda i: row[i])
            row[index] += step
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise ValueError("integer routing selections cannot be conserved")


def integer_counts(global_rows, experts=384, topk=6, ep=8, seed=0):
    if not global_rows:
        return tuple((0,) * (experts // ep) for _ in range(ep))
    # CPU torch.rand(float32) uses the low24 bits of MT19937 uint32 draws.
    integers = np.random.RandomState(seed).randint(0, 2**32, size=experts, dtype=np.uint32)
    uniforms = (integers & np.uint32(2**24 - 1)).astype(np.float32) / np.float32(2**24)
    low, high = (1, global_rows * 0.8) if global_rows * topk > experts else (0.01, 2)
    exponent = 1 - 1.01
    weights = np.power(np.float32(high**exponent - low**exponent) * uniforms + np.float32(low**exponent),
                       np.float32(1 / exponent))
    values = np.rint(weights / weights.sum(dtype=np.float32) * np.float32(global_rows * topk)).astype(np.int64)
    overflow = int(np.maximum(values - global_rows, 0).sum())
    values = np.minimum(values, global_rows)
    counts = values.reshape(ep, experts // ep).tolist()
    if overflow:
        adjust(counts, overflow, 1, global_rows)
    difference = global_rows * topk - sum(sum(row) for row in counts)
    if difference:
        adjust(counts, abs(difference), 1 if difference > 0 else -1, global_rows)
    heaviest = max(range(ep), key=lambda rank: sum(counts[rank]))
    counts[0], counts[heaviest] = counts[heaviest], counts[0]
    assert sum(sum(row) for row in counts) == global_rows * topk
    assert all(0 <= n <= global_rows for row in counts for n in row)
    return tuple(tuple(row) for row in counts)
