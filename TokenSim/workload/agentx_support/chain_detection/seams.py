# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve context-compaction continuations and rekey remaining forks."""

from __future__ import annotations

import heapq
import numpy as np
from .records import _EPSILON_SECONDS, _NormalRequestT, _req_end, _np_lcp, AgentChain


def _last_hash_outer_idx(chain: AgentChain) -> int | None:
    """Outer index of the chain's last HASH-BEARING request.

    Empty-hash rows carry no LCP evidence and must stay invisible to
    detection (spec §8), so they cannot make a dead tail look "extended"."""
    for oi, req in reversed(chain.requests):
        if req.hash_ids:
            return oi
    return None


def _elect_continuation(
    chains: list[AgentChain],
    registered: list[int],
    t_req: _NormalRequestT,
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int | None:
    """Pick the seam continuation among forks registered on a dead tail.

    Eligibility: positive depth, no temporal overlap with the tail, same
    model. Election: deepest LCP, tie-break earliest fork_time, then lowest
    chain index.

    Seam guard (``max_gap_seconds`` / ``min_overlap_ratio``): a genuine context
    compaction continues promptly, so a candidate that starts more than
    ``max_gap_seconds`` after the tail ends AND shares less than
    ``min_overlap_ratio`` of the tail's blocks is rejected -- it is a distinct
    session that merely reuses the base prefix, and stitching it on would
    fabricate a multi-hour intra-conversation idle gap. Both conditions must
    hold, so prompt compactions (any overlap) and verbatim long-gap resumes
    (high overlap) still elect."""
    t_end = _req_end(t_req)
    tail_blocks = len(t_req.hash_ids)

    def _seam_blocked(ci: int) -> bool:
        if tail_blocks == 0:
            return False
        gap = chains[ci].requests[0][1].t - t_end
        overlap = chains[ci].fork.depth / tail_blocks
        return gap > max_gap_seconds and overlap < min_overlap_ratio

    candidates = [
        ci
        for ci in registered
        if chains[ci].fork is not None
        and chains[ci].fork.depth > 0
        and t_end <= chains[ci].requests[0][1].t + _EPSILON_SECONDS
        and chains[ci].requests[0][1].model == t_req.model
        and not _seam_blocked(ci)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda ci: (
            chains[ci].fork.depth,
            -chains[ci].fork.fork_time,
            -ci,
        ),
    )


def _rekey_leftover_forks(
    *,
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    req_by_outer: dict[int, _NormalRequestT],
    registered: list[int],
    elected: int,
    owner: int,
    new_tail_outer: int,
) -> bool:
    """Re-evaluate non-elected forks against the merged chain's new tail.

    A fork's LCP relationship is to the evolving conversation state, not to
    one frozen snapshot — without re-keying, phase-1's deepest-tail
    tie-break can register a later compaction continuation under an
    already-consumed tail and strand it as a spurious spawn. Returns True
    when any fork was re-keyed (the caller re-arms that tail's key)."""
    new_tail_hash = np.asarray(req_by_outer[new_tail_outer].hash_ids, np.uint64)
    rekeyed = False
    for ci in registered:
        c = chains[ci]
        if ci == elected or c.spliced_into is not None or c.fork is None:
            continue
        d = _np_lcp(
            new_tail_hash, np.asarray(c.requests[0][1].hash_ids, dtype=np.uint64)
        )
        if d <= 0:
            continue
        c.fork.fork_outer_idx = new_tail_outer
        c.fork.depth = d
        c.fork.parent_chain = owner
        forks_by_tail.setdefault(new_tail_outer, []).append(ci)
        rekeyed = True
    return rekeyed


def _resolve_seams(
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    chain_of_request: dict[int, int],
    req_by_outer: dict[int, _NormalRequestT],
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int:
    """Phase 2: splice join-seam continuations onto dead tails.

    For each fork-source request T in time order: if T is still the final
    hash-bearing request of its (post-splice) chain, elect among its
    temporally-feasible same-model forks the deepest one (tie: earliest
    fork_time, then lowest index) as the same agent's continuation and
    splice it on. Everything else stays a spawn.

    Cascades: a splice moves the chain's tail to a later request, whose own
    forks are visited later in the worklist, and the non-elected forks of
    the consumed tail are re-keyed against the merged chain's new tail (see
    :func:`_rekey_leftover_forks`)."""
    alias: dict[int, int] = {}

    def _resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    seams = 0
    keys = sorted(forks_by_tail)
    heapq.heapify(keys)
    processed: set[int] = set()
    while keys:
        fork_outer_idx = heapq.heappop(keys)
        if fork_outer_idx in processed:
            continue
        processed.add(fork_outer_idx)
        owner = _resolve(chain_of_request[fork_outer_idx])
        owner_chain = chains[owner]
        if _last_hash_outer_idx(owner_chain) != fork_outer_idx:
            continue  # longer state was extended -> all forks are spawns
        registered = [
            ci
            for ci in forks_by_tail[fork_outer_idx]
            if chains[ci].spliced_into is None
        ]
        elected = _elect_continuation(
            chains,
            registered,
            req_by_outer[fork_outer_idx],
            max_gap_seconds=max_gap_seconds,
            min_overlap_ratio=min_overlap_ratio,
        )
        if elected is None:
            continue
        target = chains[elected]
        owner_chain.requests.extend(target.requests)
        for oi, _ in target.requests:
            chain_of_request[oi] = owner
        target.spliced_into = owner
        alias[elected] = owner
        seams += 1

        new_tail_outer = _last_hash_outer_idx(owner_chain)
        if new_tail_outer is None:
            continue
        if _rekey_leftover_forks(
            chains=chains,
            forks_by_tail=forks_by_tail,
            req_by_outer=req_by_outer,
            registered=registered,
            elected=elected,
            owner=owner,
            new_tail_outer=new_tail_outer,
        ):
            # The new tail's key may sort below the current one or may have
            # been visited already as a no-op; allow it to (re)process so
            # the re-keyed candidates get their election.
            processed.discard(new_tail_outer)
            heapq.heappush(keys, new_tail_outer)
    return seams
