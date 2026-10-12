# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flattened-agent chain detection for Weka traces via hash_id LCP evidence.

Untagged agent fan-outs (Workflow agents, headerless subagents) are recorded
as interleaved flat top-level requests. This module partitions a trace's
top-level requests back into per-agent chains and classifies every hash-list
divergence as either a join seam (the same agent continuing after a context
edit such as compaction) or a spawn (a new agent forked from a shared
prefix).

Seam-vs-spawn rule: on a shrink — a request sharing only a proper prefix of
its best-matching chain tail — the request is the same agent's continuation
(join seam) iff no future request ever pulls back to the longer, pre-shrink
state; otherwise it is a separate spawned agent. Implemented offline:
phase 1 builds chains greedily, forking on every shrink; phase 2 splices the
elected continuation back onto tails whose longer state turned out to be
dead. A deeper sibling fork from the same tail IS the "future pullback"
that demotes shallower forks to spawns, so the election encodes the rule
directly.

Same-model rule: a chain is only ever continued — by extension or by seam
splice — by requests of the SAME model. Cross-model attachment is always a
spawn, even on a full-prefix match (a Haiku worker reading Opus context is
a different agent; the rare mid-session /model switch is accepted as a
split).
"""

from __future__ import annotations

# Adapted from NVIDIA AIPerf 49634d1c. The chain algorithm and defaults are
# retained; imports target TokenSim records, and unsigned hash storage also
# preserves valid hash IDs above 2**63. No AIPerf runtime dependency.

from TokenSim.workload.agentx_support.chain_detection.records import (
    _EPSILON_SECONDS,
    _NormalRequestT,
    IndexedRequest,
    _req_end,
    _np_lcp,
    ChainFork,
    AgentChain,
    ChainDetectionResult,
    is_aux_chain,
    is_reduction_chain,
)
from TokenSim.workload.agentx_support.chain_detection.groups import (
    _overlap_components,
    worker_group_assignment,
)
from TokenSim.workload.agentx_support.chain_detection.detection import (
    _Phase1State,
    detect_agent_chains,
    _find_extension_target,
    _max_lcp_chain,
)
from TokenSim.workload.agentx_support.chain_detection.seams import (
    _last_hash_outer_idx,
    _elect_continuation,
    _rekey_leftover_forks,
    _resolve_seams,
)
