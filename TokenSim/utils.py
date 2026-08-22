from __future__ import annotations
import random
import numpy as np

from TokenSim.errors import ConfigurationError


def get_prefill_lens(len_mean: int, len_range: int, request_count: int) -> list[int]:
    if len_range < 0:
        return [len_mean] * request_count
    low = len_mean - (len_range // 2)
    high = len_mean + (len_range // 2)
    prefill_lens = list(map(lambda _: random.randint(low, high), range(request_count)))
    return prefill_lens


def get_decode_lens(
    distribution: str, len_mean: int, len_range: int, request_count: int
) -> list[int]:
    if len_range <= 0:
        return [len_mean] * request_count
    if distribution == "uniform":
        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        decode_lens = list(map(lambda _: random.randint(low, high), range(request_count)))
        return decode_lens
    elif distribution == "exponential":
        return [
            min(round(s), len_range)
            for s in np.random.exponential(scale=len_mean, size=request_count)
        ]
    elif distribution == "capped_exponential":
        response_lens = []
        while len(response_lens) < request_count:
            sample = round(np.random.exponential(scale=len_mean))
            if sample >= 2 and sample <= len_range:
                response_lens.append(sample)
        return response_lens
    elif distribution == "burst":
        return [len_mean] * request_count
    else:
        raise ConfigurationError(f"unknown decode length distribution {distribution!r}")


def get_wait_time(mean_time_between_requests: float, distribution: str) -> float:
    if distribution == "uniform":
        return mean_time_between_requests
    else:
        return np.random.exponential(mean_time_between_requests)
