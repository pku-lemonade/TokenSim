# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporal worker groups and setup prefixes shared by live chains."""

from __future__ import annotations

import heapq
from .records import _IntervalCand, _req_end, ChainDetectionResult


def _overlap_components(cands: list[_IntervalCand]) -> list[list[_IntervalCand]]:
    """Connected components of ``[t0, t1)`` interval overlap (sweep + union-find).

    Processing in start order, every interval still "active" (end > this start)
    overlaps the current one; the active set is always one component, so unioning
    with any active member joins the whole component. Touching intervals
    (``end == start``) do not overlap."""
    if not cands:
        return []
    order = sorted(range(len(cands)), key=lambda i: (cands[i][0], cands[i][2]))
    parent = list(range(len(cands)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active: list[tuple[float, int]] = []  # min-heap of (t1, idx)
    for i in order:
        t0 = cands[i][0]
        while active and active[0][0] <= t0:
            heapq.heappop(active)
        if active:
            parent[_find(i)] = _find(active[0][1])
        heapq.heappush(active, (cands[i][1], i))

    comps: dict[int, list[_IntervalCand]] = {}
    for i in range(len(cands)):
        comps.setdefault(_find(i), []).append(cands[i])
    return list(comps.values())


def worker_group_assignment(
    result: ChainDetectionResult,
    *,
    group_min: int,
) -> dict[int, tuple[int, int]]:
    """Assign each worker-group member a ``(group, member)`` coordinate.

    A parallel sub-agent fan-out is a set of persistent workers that share a deep
    spawned context AND run CONCURRENTLY. We require BOTH, the way the corpus
    analysis and the graph adapter's sibling-clustering prescribe (overlapping
    intervals AND a shared prefix):

    1. **Shared-context scope**: bucket workers that forked from shared context
       (``fork.depth > 0``) by their fork point
       (``fork.parent_chain`` + ``fork.fork_outer_idx``) -- workers off the same
       parent request share that parent's deep prefix. This scope keeps unrelated
       fan-outs apart; without it, pure interval overlap bridges every worker in
       a busy trace into one blob (a long-running chain of overlaps spans the
       whole session).
    2. **Temporal overlap within the scope**: within each fork-point bucket,
       split into connected components of overlapping ``[t0, t1)`` intervals
       (``t0`` = first request time, ``t1`` = latest request-interval end). This
       drops members that share the fork point but never actually run
       concurrently (e.g. a chain seam-re-keyed onto a parent tail it never ran
       beside) -- the group is a concurrent swarm, not a session-wide lineage
       (the failure of keying on ``hash_ids[0]``).

    - **group**: a (fork-point, overlap-component) with at least ``group_min``
      members, numbered by earliest start.
    - **member**: index within the group, in ``(t0, outer_idx)`` order.

    Returns ``{chain_index: (group, member)}`` for members only (a subset of
    ``result.worker_indices``); empty when ``group_min <= 0``."""
    if group_min <= 0:
        return {}

    buckets: dict[tuple[int, int], list[_IntervalCand]] = {}
    for ci in result.worker_indices:
        chain = result.chains[ci]
        fork = chain.fork
        if (
            fork is None
            or fork.depth <= 0
            or fork.parent_chain is None
            or fork.fork_outer_idx is None
            or not chain.requests
        ):
            continue
        t0 = chain.requests[0][1].t
        t1 = max(_req_end(req) for _, req in chain.requests)
        key = (fork.parent_chain, fork.fork_outer_idx)
        buckets.setdefault(key, []).append((t0, t1, chain.requests[0][0], ci))

    components: list[list[_IntervalCand]] = []
    for cands in buckets.values():
        components.extend(_overlap_components(cands))

    groups = [comp for comp in components if len(comp) >= group_min]
    groups.sort(key=lambda comp: min((c[0], c[2]) for c in comp))

    out: dict[int, tuple[int, int]] = {}
    for group, comp in enumerate(groups):
        comp.sort(key=lambda c: (c[0], c[2]))
        for member, (_, _, _, ci) in enumerate(comp):
            out[ci] = (group, member)
    return out
