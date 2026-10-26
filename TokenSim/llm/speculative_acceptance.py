"""Stateless acceptance draws keyed by the committed verification round."""

from hashlib import blake2b
import math


def sample_acceptance(config, mean, seed, request_id, round_index):
    def uniform(position):
        key = f"{seed}:{request_id}:{round_index}:{position}".encode("ascii")
        bits = int.from_bytes(
            blake2b(key, digest_size=8, person=b"ts-mtp-round-v1").digest(), "big"
        )
        return (bits >> 11) / (1 << 53)

    if config.kind == "conditional":
        accepted = 1
        for position, rate in enumerate(config.rates):
            if uniform(position) >= rate:
                break
            accepted += 1
        return accepted
    if config.kind == "empirical":
        draw, cumulative = uniform(0), 0.0
        for accepted, probability in enumerate(config.distribution, 1):
            cumulative += probability
            if draw < cumulative:
                return accepted
        return len(config.distribution)
    low = int(math.floor(mean))
    return low + int(uniform(0) < mean - low)


def legacy_acceptance(mean, seed, request_id, generation_index):
    state = (
        request_id * 1103515245 + generation_index * 12345 + seed * 0x9E3779B9
    ) & 0xFFFFFFFF
    return int(math.floor(mean)) + int(state / 4294967296.0 < mean % 1)
