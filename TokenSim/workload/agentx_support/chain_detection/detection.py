# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Greedy chain detection and same-model tail extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from .records import (
    _EPSILON_SECONDS,
    _NormalRequestT,
    IndexedRequest,
    _req_end,
    _np_lcp,
    ChainFork,
    AgentChain,
    ChainDetectionResult,
)
from .seams import _resolve_seams


@dataclass(slots=True)
class _Phase1State:
    """Working state for the greedy forward pass."""

    chains: list[AgentChain] = field(default_factory=list)
    chain_of_request: dict[int, int] = field(default_factory=dict)
    forks_by_tail: dict[int, list[int]] = field(default_factory=dict)
    req_by_outer: dict[int, _NormalRequestT] = field(default_factory=dict)
    unclassified: int = 0

    def _append(self, chain_idx: int, outer_idx: int, req: _NormalRequestT) -> None:
        c = self.chains[chain_idx]
        c.requests.append((outer_idx, req))
        self.chain_of_request[outer_idx] = chain_idx
        if req.hash_ids:
            c.tail_outer_idx = outer_idx
            c.tail_hash = np.asarray(req.hash_ids, dtype=np.uint64)
            c.tail_end = _req_end(req)
            c.tail_model = req.model

    def classify(self, outer_idx: int, req: _NormalRequestT) -> None:
        self.req_by_outer[outer_idx] = req
        if not req.hash_ids:
            # No LCP evidence: keep on the main chain, invisible to tails
            # and forks (it must not found a chain or serve as a witness).
            self.unclassified += 1
            if not self.chains:
                self.chains.append(AgentChain())
            self.chains[0].requests.append((outer_idx, req))
            self.chain_of_request[outer_idx] = 0
            return

        h = np.asarray(req.hash_ids, dtype=np.uint64)
        if not self.chains:
            # The trace's first request founds the main chain. It forked
            # from nothing, so fork stays None (the AgentChain contract).
            self.chains.append(AgentChain())
            self._append(0, outer_idx, req)
            return
        target = _find_extension_target(self.chains, h, req.t, req.model)
        if target is not None:
            self._append(target, outer_idx, req)
            return

        parent, depth = _max_lcp_chain(self.chains, h)
        if parent is None and all(c.tail_hash.shape[0] == 0 for c in self.chains):
            # No chain carries hash evidence yet (only leading empty-hash
            # rows): this is the trace's first hash-bearing request and IS
            # the main agent — joining chain 0 keeps the empty rows "on the
            # main chain" instead of exiling all real load to a spawn.
            self._append(0, outer_idx, req)
            return
        fork = ChainFork(
            parent_chain=parent,
            fork_outer_idx=(
                self.chains[parent].tail_outer_idx if parent is not None else None
            ),
            depth=depth,
            fork_time=req.t,
        )
        new_idx = len(self.chains)
        self.chains.append(AgentChain(fork=fork))
        self._append(new_idx, outer_idx, req)
        if fork.fork_outer_idx is not None and depth > 0:
            self.forks_by_tail.setdefault(fork.fork_outer_idx, []).append(new_idx)


def detect_agent_chains(
    normals: list[IndexedRequest],
    *,
    seam_max_gap_seconds: float = 3600.0,
    seam_min_overlap_ratio: float = 0.5,
) -> ChainDetectionResult:
    """Partition retained top-level requests into per-agent chains.

    ``seam_max_gap_seconds`` / ``seam_min_overlap_ratio`` gate join-seam
    splicing (see :func:`_elect_continuation`): a low-overlap continuation far
    past the chain tail is spawned as its own conversation rather than stitched
    on, so a distinct session that merely shares a base prefix does not get
    folded into one conversation with a multi-hour internal gap. Defaults match
    ``Environment.DATASET.WEKA_SEAM_*``; the WekaTraceLoader call sites pass the
    configured values, while direct callers (tests) use these literals."""
    if not normals:
        return ChainDetectionResult(
            chains=[],
            main_index=0,
            worker_indices=[],
            seams_merged=0,
            unclassified_empty_hash=0,
        )

    # Spec §4: process in (t, outer_idx) order. Trace files are normally
    # time-ordered already; sorting makes the algorithm independent of file
    # order and keeps every chain's request list t-ordered for delay math.
    ordered = sorted(normals, key=lambda item: (item[1].t, item[0]))

    state = _Phase1State()
    for outer_idx, req in ordered:
        state.classify(outer_idx, req)
    chains = state.chains

    seams = _resolve_seams(
        chains,
        state.forks_by_tail,
        state.chain_of_request,
        state.req_by_outer,
        max_gap_seconds=seam_max_gap_seconds,
        min_overlap_ratio=seam_min_overlap_ratio,
    )

    alias = {
        i: c.spliced_into for i, c in enumerate(chains) if c.spliced_into is not None
    }

    def _resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    for c in chains:
        if (
            c.spliced_into is None
            and c.fork is not None
            and c.fork.parent_chain is not None
        ):
            c.fork.parent_chain = _resolve(c.fork.parent_chain)

    main_index = _resolve(state.chain_of_request[ordered[0][0]])
    workers = [
        i for i, c in enumerate(chains) if c.spliced_into is None and i != main_index
    ]
    workers.sort(key=lambda i: (chains[i].requests[0][1].t, chains[i].requests[0][0]))
    return ChainDetectionResult(
        chains=chains,
        main_index=main_index,
        worker_indices=workers,
        seams_merged=seams,
        unclassified_empty_hash=state.unclassified,
    )


def _find_extension_target(
    chains: list[AgentChain], h: np.ndarray, t: float, model: str
) -> int | None:
    """Chain whose tail is a complete prefix of ``h``, has ended by ``t``,
    and ran on the same ``model`` (cross-model attachment is always a spawn).

    Deepest tail wins; ties go to the lowest chain index (ascending scan
    with strict ``>`` keeps the first)."""
    best: int | None = None
    best_len = -1
    hn = h.shape[0]
    for idx, c in enumerate(chains):
        tl = c.tail_hash.shape[0]
        if tl == 0 or tl > hn or tl <= best_len:
            continue
        if c.tail_model != model:
            continue
        if c.tail_end > t + _EPSILON_SECONDS:
            continue
        if c.tail_hash[tl - 1] != h[tl - 1]:
            continue
        if bool((h[:tl] == c.tail_hash).all()):
            best, best_len = idx, tl
    return best


def _max_lcp_chain(chains: list[AgentChain], h: np.ndarray) -> tuple[int | None, int]:
    """Chain tail with the deepest LCP against ``h`` (ties: deeper tail,
    then lower index). Returns (None, 0) when nothing shares a prefix."""
    best_idx: int | None = None
    best_key = (0, 0)
    for idx, c in enumerate(chains):
        if c.tail_hash.shape[0] == 0:
            continue
        d = _np_lcp(c.tail_hash, h)
        if d == 0:
            continue
        key = (d, c.tail_hash.shape[0])
        if key > best_key:
            best_idx, best_key = idx, key
    return best_idx, best_key[0]
