# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/router v0.1.14 src/policies/consistent_hash.rs.
# MurmurHash64A/FurcHash there are attributed to Facebook mcrouter hash.c.
# Modifications: Python integer wrapping and a static, healthy-worker ring.
"""Reference router hashing for stable independent conversation placement."""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter

from TokenSim.errors import ConfigurationError

MASK = (1 << 64) - 1
M = 0xC6A4A7935BD1E995
SEED = 4193360111


def murmur_hash_64a(data: bytes, seed=SEED):
    h = seed ^ ((len(data) * M) & MASK)
    size = len(data) // 8 * 8
    for offset in range(0, size, 8):
        k = (int.from_bytes(data[offset : offset + 8], "little") * M) & MASK
        k ^= k >> 47
        k = (k * M) & MASK
        h = ((h ^ k) * M) & MASK
    if size < len(data):
        h ^= int.from_bytes(data[size:], "little")
        h = (h * M) & MASK
    h ^= h >> 47
    h = (h * M) & MASK
    return h ^ (h >> 47)


def murmur_rehash_64a(value):
    return murmur_hash_64a(value.to_bytes(8, "little"))


def fbi_hash(key: str):
    data, hashes = key.encode(), []

    def bit(index):
        order = index >> 6
        while len(hashes) <= order:
            hashes.append(
                murmur_rehash_64a(hashes[-1]) if hashes else murmur_hash_64a(data)
            )
        return (hashes[order] >> (index & 63)) & 1

    modulus = (1 << 23) - 1
    depth = 23
    index = depth
    result = 0
    for _ in range(32):
        while not bit(index):
            if depth == 0:
                return murmur_hash_64a((0).to_bytes(4, "little"))
            depth -= 1
            index = depth
        index += 23
        number = 1
        for _ in range(max(depth - 1, 0)):
            number = (number << 1) | bit(index)
            index += 23
        if number < modulus:
            result = number
            break
    return murmur_hash_64a(result.to_bytes(4, "little"))


class ConsistentHashRouter:
    """160 virtual nodes per URL, including each DP-aware @rank suffix.

    The benchmark aliases X-Correlation-ID into X-Session-ID. Parent and child
    conversations supply their own session identity; the parent ID is not used.
    """

    def __init__(self, workers, *, backend_url="http://localhost:8889"):
        self.workers = {w.dp_rank: w for w in workers}
        if len(self.workers) != len(workers) or not workers:
            raise ConfigurationError(
                "router needs exactly one schedulable leader per DP rank"
            )
        self.urls = {rank: f"{backend_url}@{rank}" for rank in self.workers}
        ring = {}
        for rank, url in self.urls.items():
            for ordinal in range(160):
                ring[fbi_hash(f"{url}:{ordinal}")] = rank
        self.positions = sorted(ring)
        self.ranks = [ring[position] for position in self.positions]
        self.sessions = {}
        self.request_counts = Counter()
        self.session_counts = Counter()
        self.backend_url = backend_url

    def rank_for_key(self, hash_key):
        index = bisect_left(self.positions, fbi_hash(hash_key))
        return self.ranks[index % len(self.ranks)]

    def select(self, request):
        session = request.chat_id
        if not session:
            raise ConfigurationError(
                "consistent-hash routing requires a conversation/session ID"
            )
        if session not in self.sessions:
            self.sessions[session] = self.rank_for_key(f"header:x-session-id:{session}")
            self.session_counts[self.sessions[session]] += 1
        rank = self.sessions[session]
        request.dp_rank = rank
        request.routing_worker_id = self.workers[rank].id
        self.request_counts[rank] += 1
        return self.workers[rank]

    def reset_stats(self):
        self.request_counts.clear()
        self.session_counts.clear()

    def describe(self):
        return {
            "policy": "consistent_hash",
            "reference_version": "0.1.14",
            "hash": "FurcHash plus MurmurHash64A",
            "virtual_nodes_per_worker": 160,
            "backend_url": self.backend_url,
            "requests_per_rank": dict(self.request_counts),
            "new_sessions_per_rank": dict(self.session_counts),
            "affinity_sessions_retained": len(self.sessions),
        }


class FixedWorkerRouter:
    """Replay a verified historical conversation-to-rank mapping.

    Unknown conversations deliberately fall back to a normal consistent hash
    and are counted. This keeps the diagnostic runnable without fabricating a
    rank while making coverage explicit in ``describe``.
    """

    def __init__(self, workers, mapping, *, backend_url="http://localhost:8889"):
        self.workers = {w.dp_rank: w for w in workers}
        self.mapping = {str(key): int(value) for key, value in mapping.items()}
        if any(rank not in self.workers for rank in self.mapping.values()):
            raise ConfigurationError(
                "fixed worker mapping references an unknown DP rank"
            )
        self.fallback = ConsistentHashRouter(workers, backend_url=backend_url)
        self.request_counts = Counter()
        self.mapped_request_counts = Counter()
        self.missing_keys = Counter()
        self.session_counts = Counter()

    def select(self, request):
        key = getattr(request, "agentx_conversation_id", None) or request.chat_id
        rank = self.mapping.get(str(key))
        if rank is None:
            self.missing_keys[str(key)] += 1
            worker = self.fallback.select(request)
            rank = worker.dp_rank
        else:
            request.dp_rank = rank
            worker = self.workers[rank]
            request.routing_worker_id = worker.id
            self.mapped_request_counts[rank] += 1
        self.request_counts[rank] += 1
        self.session_counts[rank] += 1
        return worker

    def reset_stats(self):
        self.request_counts.clear()
        self.mapped_request_counts.clear()
        self.missing_keys.clear()
        self.session_counts.clear()
        self.fallback.reset_stats()

    def describe(self):
        return {
            "policy": "fixed_worker_mapping",
            "fallback_policy": "consistent_hash",
            "mapping_entries": len(self.mapping),
            "requests_per_rank": dict(self.request_counts),
            "mapped_requests_per_rank": dict(self.mapped_request_counts),
            "missing_key_count": sum(self.missing_keys.values()),
            "missing_key_examples": list(self.missing_keys)[:20],
            "fallback": self.fallback.describe(),
        }
