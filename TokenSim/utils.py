from __future__ import annotations
import random
import numpy as np

from TokenSim.errors import ConfigurationError


def get_prefill_lens(
    len_mean: int, len_range: int, request_count: int, *, rng=None
) -> list[int]:
    rng = rng or random
    if len_range < 0:
        return [len_mean] * request_count
    low = len_mean - (len_range // 2)
    high = len_mean + (len_range // 2)
    prefill_lens = [rng.randint(low, high) for _ in range(request_count)]
    return prefill_lens


def get_decode_lens(
    distribution: str,
    len_mean: int,
    len_range: int,
    request_count: int,
    *,
    rng=None,
    numpy_rng=None,
) -> list[int]:
    rng = rng or random
    numpy_rng = numpy_rng if numpy_rng is not None else np.random
    if len_range <= 0:
        return [len_mean] * request_count
    if distribution == "uniform":
        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        decode_lens = list(rng.randint(low, high) for _ in range(request_count))
        return decode_lens
    elif distribution == "exponential":
        return [
            max(1, min(round(s), len_range))
            for s in numpy_rng.exponential(scale=len_mean, size=request_count)
        ]
    elif distribution == "capped_exponential":
        response_lens = []
        while len(response_lens) < request_count:
            sample = round(numpy_rng.exponential(scale=len_mean))
            if sample >= 2 and sample <= len_range:
                response_lens.append(sample)
        return response_lens
    elif distribution == "burst":
        return [len_mean] * request_count
    else:
        raise ConfigurationError(f"unknown decode length distribution {distribution!r}")


class BoundedCache(dict):
    """Capacity-bound memo for pure computations on immutable inputs.

    Clear on capacity pressure; callers account for queries separately from
    the cached computation, so eviction cannot change simulation statistics.
    """

    def __init__(self, max_entries: int):
        if max_entries <= 0:
            raise ValueError("cache capacity must be positive")
        super().__init__()
        self.max_entries = max_entries

    def __setitem__(self, key, value):
        if len(self) >= self.max_entries and key not in self:
            self.clear()
        super().__setitem__(key, value)
