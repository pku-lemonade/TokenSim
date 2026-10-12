# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chain records, hash-prefix evidence and auxiliary-chain classification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
from TokenSim.workload.agentx_support.models import RawRequest


WekaNormalRequest = WekaStreamingRequest = RawRequest

_EPSILON_SECONDS = 1e-6

_NormalRequestT = WekaNormalRequest | WekaStreamingRequest

IndexedRequest = tuple[int, _NormalRequestT]

_IntervalCand = tuple[float, float, int, int]  # (t0, t1, first_outer, chain_index)


def _req_end(req: _NormalRequestT) -> float:
    """Interval end in seconds; missing/negative/non-finite api_time counts as zero.

    The finite check matters for temporal feasibility: a NaN or +inf
    ``api_time`` would otherwise poison a chain tail's end and either block
    every same-context extension forever (+inf) or make comparisons
    unreliable (NaN).
    """
    duration = (
        req.api_time
        if req.api_time is not None and math.isfinite(req.api_time)
        else 0.0
    )
    return req.t + max(duration, 0.0)


def _np_lcp(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common prefix of two int64 hash arrays."""
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    neq = a[:n] != b[:n]
    i = int(neq.argmax())
    return i if neq[i] else n


@dataclass(slots=True)
class ChainFork:
    """Where a chain split off another chain."""

    parent_chain: int | None
    """Phase-1 index of the chain forked from; None = no shared context.
    Rewritten to the live (post-splice) chain index by phase 2."""
    fork_outer_idx: int | None
    """Outer index of the tail request T this chain forked from."""
    depth: int
    """Blocks shared with T at fork time (LCP)."""
    fork_time: float
    """t of this chain's first request."""


@dataclass(slots=True)
class AgentChain:
    """One detected agent: a time-ordered run of requests."""

    requests: list[IndexedRequest] = field(default_factory=list)
    """The chain's requests in (t, outer_idx) order."""
    fork: ChainFork | None = None
    """How this chain came to exist; None only for the first chain."""
    spliced_into: int | None = None
    """Set by phase 2 when this chain was a join-seam continuation."""
    tail_outer_idx: int = -1
    """Outer index of the last hash-bearing request (phase-1 state)."""
    tail_hash: np.ndarray = field(default_factory=lambda: np.empty(0, np.uint64))
    """Hash array of the last hash-bearing request (phase-1 state)."""
    tail_end: float = 0.0
    """Interval end of the last hash-bearing request (phase-1 state)."""
    tail_model: str = ""
    """Model of the last hash-bearing request (phase-1 state). A chain is
    only ever continued by same-model requests; cross-model = spawn."""


@dataclass(slots=True)
class ChainDetectionResult:
    """Output of detect_agent_chains. Spliced chains stay in ``chains`` for
    fork-history logging but are excluded from ``worker_indices``."""

    chains: list[AgentChain]
    """All phase-1 chains, including spliced (dead) ones."""
    main_index: int
    """Index of the chain owning the trace's first retained request."""
    worker_indices: list[int]
    """Live non-main chains, ordered by first request (t, outer_idx)."""
    seams_merged: int
    """Number of join-seam splices performed in phase 2."""
    unclassified_empty_hash: int
    """Requests with empty hash_ids kept on the main chain as-is."""


def is_aux_chain(
    requests: list[_NormalRequestT],
    main_peak_isl: int,
    *,
    max_requests: int,
    isl_ratio: float,
    isl_floor: int,
    main_model: str | None = None,
    cross_model: bool = True,
) -> bool:
    """Classify a detected worker chain as an auxiliary one-shot call.

    A detected worker chain is reclassified as *auxiliary* -- a sidecar rather
    than a genuine agent -- when it is short and either (a) starts from a small,
    fresh context, or (b) runs on a different model than the enclosing agent.
    Both are signatures of a tool-issued one-shot (web fetch / search summary,
    title generation, a classifier) that rides alongside the agent for a single
    call, not a sustained sub-agent. Applied at both layers -- top-level flat
    chains (genuine ``::fa:`` vs sidecar ``::aux:``) and a subagent's nested-LCP
    overflow chains (genuine ``:fa:`` vs sidecar ``:aux:``).

    ``requests`` is the chain's request list (time-ordered). Returns ``True``
    iff it holds at most ``max_requests`` requests AND either:

    - **cross-model** (``cross_model`` set and ``main_model`` given): the first
      request runs on a model other than the enclosing main chain's. An agent
      does not switch models for its own reasoning, so a one-shot on a different
      model is a tool-internal call (e.g. a Haiku ``WebFetch`` summary under an
      Opus agent) regardless of payload size; or
    - **small fresh context**: the first request's input length is below
      ``max(isl_floor, isl_ratio * main_peak_isl)`` -- small both absolutely and
      relative to the surrounding context. ``main_peak_isl`` is the largest
      input length on the enclosing main chain (the trace's for flat chains, the
      subagent's for overflow).

    Set ``max_requests=0`` to disable reclassification entirely (every worker
    chain keeps its agent tag); ``cross_model=False`` disables only arm (a).
    """
    if not requests or len(requests) > max_requests:
        return False
    first = requests[0]
    if cross_model and main_model is not None and first.model != main_model:
        return True
    threshold = max(isl_floor, int(isl_ratio * main_peak_isl))
    return first.input_length < threshold


def is_reduction_chain(
    requests: list[_NormalRequestT],
    *,
    osl_max: int,
    ratio: float,
    isl_floor: int,
) -> bool:
    """Classify a single-request worker chain as an auxiliary *reduction* call.

    A reduction is a large-input / short-output one-shot -- a context
    compaction, a subagent-result summary, or a tool-output digest -- that
    consumes a large body and emits a short summary. It runs on the SAME model
    as the enclosing agent (so :func:`is_aux_chain`'s cross-model arm misses it)
    and starts from a large context (so the small-fresh-context size arm misses
    it too), yet it is auxiliary, not generative: a real agent emits long
    completions, while a reduction's output stays bounded and its input/output
    ratio is high. The signature is stable across every weka capture (the one
    flat-chain class present from the earliest snapshot onward).

    Returns ``True`` iff ``requests`` is exactly one request whose output length
    is in ``(0, osl_max)``, whose input length is at least ``isl_floor``, and
    whose input/output ratio exceeds ``ratio``. Set ``osl_max=0`` to disable.
    Intended to be checked only AFTER :func:`is_aux_chain` returns False, so a
    cross-model or small-fresh-context one-shot is never relabeled a reduction.
    """
    if osl_max <= 0 or len(requests) != 1:
        return False
    first = requests[0]
    osl = first.output_length or 0
    if osl <= 0 or osl >= osl_max:
        return False
    if first.input_length < isl_floor:
        return False
    return first.input_length > ratio * osl
