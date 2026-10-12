# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf-equivalent conversation splitting without text synthesis or SDK imports.

Rules are derived from AIPerf 49634d1c weka_trace.py and
timing/replay_dependencies.py. The retained chain classifier has its own
Apache-2.0 attribution; this module builds TokenSim's minimal metadata graph.
"""

from __future__ import annotations

import json
import math
from array import array
from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from TokenSim.errors import WorkloadValidationError
from TokenSim.workload.agentx_support.chains import (
    detect_agent_chains,
    is_aux_chain,
    is_reduction_chain,
    worker_group_assignment,
)
from TokenSim.workload.agentx_support.models import (
    Branch,
    Conversation,
    GraphTurn,
    RawRequest,
    SourceKey,
    TraceGraph,
    TurnKey,
)

EPSILON = 1e-6

try:  # orjson decodes the multi-GB traces several times faster.
    from orjson import loads as _loads
except ImportError:  # pragma: no cover - optional dependency
    _loads = json.loads


def _number(value, name: str, *, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkloadValidationError(f"invalid AgentX {name}: {value!r}") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise WorkloadValidationError(f"invalid AgentX {name}: {value!r}")
    return number


def _integer(value, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise WorkloadValidationError(f"invalid AgentX {name}: {value!r}")
    return value


def _hash_array(hashes) -> array:
    if not isinstance(hashes, list):
        raise WorkloadValidationError("AgentX hash_ids must be a list")
    # Exact int type excludes bool; array("Q") rejects values outside u64.
    if not set(map(type, hashes)) <= {int}:
        raise WorkloadValidationError(
            "AgentX hash IDs must be unsigned 64-bit integers"
        )
    try:
        return array("Q", hashes)
    except OverflowError as exc:
        raise WorkloadValidationError(
            "AgentX hash IDs must be unsigned 64-bit integers"
        ) from exc


def _raw_request(row: dict) -> RawRequest:
    hashes = _hash_array(row.get("hash_ids", []))
    if row.get("type") not in ("n", "s"):
        raise WorkloadValidationError(
            f"unknown AgentX request type: {row.get('type')!r}"
        )
    return RawRequest(
        t=_number(row["t"], "timestamp", nonnegative=True),
        input_length=_integer(row["in"], "input length"),
        output_length=_integer(row["out"], "output length"),
        hash_ids=hashes,
        model=str(row["model"]),
        api_time=float(row["api_time"]) if row.get("api_time") is not None else None,
        think_time=(
            float(row["think_time"]) if row.get("think_time") is not None else None
        ),
        type=row["type"],
    )


def _lcp(a, b) -> int:
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    return min(len(a), len(b))


def _split_preamble(indexed: list[tuple[int, RawRequest]]):
    # weka_trace._split_off_preamble: only the earliest disjoint request can
    # be peeled, then reattached to the root for replay. It still has identity.
    if len(indexed) < 2:
        return [], indexed
    ordered = sorted(indexed, key=lambda item: (item[1].t, item[0]))
    first, rest = ordered[0], ordered[1:]
    req = first[1]
    if not req.hash_ids or any(
        _lcp(req.hash_ids, other.hash_ids) for _, other in rest if other.hash_ids
    ):
        return [], indexed
    if req.output_length > 64:
        own = set(req.hash_ids)
        if any(not own.isdisjoint(other.hash_ids) for _, other in rest):
            return [], indexed
    return [first], sorted(rest, key=lambda item: item[0])


def _split_chains(indexed: list[tuple[int, RawRequest]], *, nested: bool = False):
    if not indexed:
        return [(None, [], False)], {"seams_merged": 0, "empty_hash_requests": 0}
    preamble, retained = _split_preamble(indexed)
    detection = detect_agent_chains(
        retained, seam_max_gap_seconds=3600.0, seam_min_overlap_ratio=0.5
    )
    detected_main = detection.chains[detection.main_index].requests
    # Top-level loader preserves original order when no workers were detected.
    main = (
        sorted(preamble + detected_main, key=lambda item: (item[1].t, item[0]))
        if nested or detection.worker_indices
        else indexed
    )
    result = [(None, main, False)]
    main_peak = max((r.input_length for _, r in detected_main), default=0)
    main_model = detected_main[0][1].model if detected_main else None
    worker_groups = worker_group_assignment(detection, group_min=3)
    for n, ci in enumerate(detection.worker_indices):
        chain = detection.chains[ci]
        requests = [r for _, r in chain.requests]
        auxiliary = is_aux_chain(
            requests,
            main_peak,
            max_requests=1,
            isl_ratio=0.1,
            isl_floor=16384,
            main_model=main_model,
            cross_model=True,
        )
        reduction = not auxiliary and is_reduction_chain(
            requests, osl_max=4000, ratio=20.0, isl_floor=16384
        )
        coord = worker_groups.get(ci)
        suffix = (
            f"aux:{n:03d}"
            if auxiliary or reduction
            else (
                f"wg:{coord[0]:03d}_{coord[1]:03d}"
                if coord is not None
                else f"fa:{n:03d}"
            )
        )
        result.append((suffix, list(chain.requests), auxiliary or reduction))
    return result, {
        "seams_merged": detection.seams_merged,
        "empty_hash_requests": detection.unclassified_empty_hash,
    }


def _conversation(
    sid: str,
    trace_id: str,
    indexed: list[tuple[int, RawRequest]],
    source_kind: str,
    *,
    outer: int | None = None,
    auxiliary: bool = False,
) -> Conversation:
    turns = []
    for turn_index, (index, req) in enumerate(indexed):
        previous = indexed[turn_index - 1][1] if turn_index else None
        # _end_to_start_delay_ms uses finite recorded API time as-is (while
        # interval inference clamps negative duration to zero). Keep that
        # distinction for malformed-duration boundary cases too.
        api_ms = (
            previous.api_time * 1000.0
            if previous
            and previous.api_time is not None
            and math.isfinite(previous.api_time)
            else 0.0
        )
        wait = (
            max(0.0, req.t * 1000.0 - previous.t * 1000.0 - api_ms) / 1000.0
            if previous
            else 0.0
        )
        source = SourceKey(
            trace_id,
            index if outer is None else outer,
            None if outer is None else index,
        )
        turns.append(
            GraphTurn(TurnKey(sid, turn_index), source, req, source_kind, wait)
        )
    return Conversation(
        sid,
        None if sid == trace_id else trace_id,
        0 if sid == trace_id else 1,
        turns,
        auxiliary,
    )


def infer_predecessors(
    conversations: list[Conversation],
) -> dict[TurnKey, tuple[TurnKey, ...]]:
    """AIPerf's recorded completion frontier, within each captured-agent scope.

    Equal starts are unordered, including zero-width intervals. Boundary
    touches are ordered. Arithmetic is in milliseconds, matching AIPerf.
    Index completed intervals by end/start/key to avoid quadratic rescans;
    the latest feasible candidate is identical to the reference max() rule.
    """
    by_scope = defaultdict(dict)
    for conversation in conversations:
        intervals = [
            (
                turn.request.t * 1000.0 + turn.request.duration_s * 1000.0,
                turn.request.t * 1000.0,
                turn.key,
                turn,
            )
            for turn in conversation.turns
        ]
        intervals.sort(key=lambda item: item[:3])
        by_scope[conversation.replay_scope_id][conversation.conversation_id] = (
            [item[0] for item in intervals],
            intervals,
        )
    result = {}
    for conversation in conversations:
        for target in conversation.turns:
            start = target.request.t * 1000.0
            frontier = []
            for sid, (ends, candidates) in by_scope[
                conversation.replay_scope_id
            ].items():
                if sid == conversation.conversation_id:
                    continue
                index = bisect_right(ends, start) - 1
                # The only end<=start intervals whose start is not earlier
                # are zero-width intervals at exactly this same timestamp.
                while index >= 0 and candidates[index][1] >= start:
                    index -= 1
                if index >= 0:
                    frontier.append(candidates[index])
            predecessors = [
                c[2]
                for c in frontier
                if not any(
                    c[1] < later[1] and c[0] <= later[1]
                    for later in frontier
                    if later is not c
                )
            ]
            result[target.key] = tuple(sorted(predecessors))
    return result


def parse_trace_graph(raw: dict) -> TraceGraph:
    trace_id = str(raw["id"])
    block_size = _integer(raw["block_size"], "block size", positive=True)
    if raw.get("hash_id_scope") != "local":
        raise WorkloadValidationError(
            "AgentX graph parser requires per-trace local hash scope"
        )
    normals, markers, original_sources = [], [], set()
    raw_count = 0
    for outer, entry in enumerate(raw["requests"]):
        if entry["type"] in ("n", "s"):
            normals.append((outer, _raw_request(entry)))
            original_sources.add(SourceKey(trace_id, outer))
            raw_count += 1
        elif entry["type"] == "subagent":
            spawn = _number(entry["t"], "subagent timestamp", nonnegative=True)
            inner = []
            for index, row in enumerate(entry["requests"]):
                req = _raw_request(row)
                if req.t + EPSILON < spawn:
                    req = replace(req, t=spawn + req.t)
                inner.append((index, req))
                original_sources.add(SourceKey(trace_id, outer, index))
                raw_count += 1
            duration = entry.get("duration_ms")
            if duration is not None:
                value = float(duration)
                end = spawn + max(0.0, value) / 1000 if math.isfinite(value) else spawn
            else:
                end = max((r.end_s for _, r in inner), default=spawn)
            markers.append((outer, entry["agent_id"], spawn, end, inner))
        else:
            raise WorkloadValidationError(
                f"unknown AgentX entry type: {entry['type']!r}"
            )
    if not normals:
        raise WorkloadValidationError(f"AgentX trace {trace_id} has no root requests")
    chains, stats = _split_chains(normals)
    root = _conversation(trace_id, trace_id, chains[0][1], "weka_main")
    conversations = [root]
    branch_groups = defaultdict(list)
    # Groups are keyed separately for tagged subagents and flat chains, as in
    # AIPerf. Each item carries stable identity, start/end, and its children.
    seen_agents = set()

    def group_key(first_outer: int, end: float, *, flat: bool):
        preceding = max(
            (
                i
                for i, turn in enumerate(root.turns)
                if turn.source.outer_index < first_outer
            ),
            default=0 if flat else None,
        )
        if preceding is None:
            raise WorkloadValidationError(
                f"orphan subagent in {trace_id}: preserving original requests requires a root predecessor"
            )
        join = next(
            (
                i
                for i, turn in sorted(
                    enumerate(root.turns), key=lambda it: it[1].source.outer_index
                )
                if turn.source.outer_index > first_outer
                and turn.request.t + EPSILON >= end
            ),
            None,
        )
        return preceding, join

    for outer, agent_id, start, end, inner in markers:
        if agent_id in seen_agents:
            raise WorkloadValidationError(
                f"duplicate AgentX subagent ID {agent_id!r} in {trace_id}"
            )
        seen_agents.add(agent_id)
        child_chains, child_stats = _split_chains(inner, nested=True)
        for key, value in child_stats.items():
            stats[key] = stats.get(key, 0) + value
        child_ids = []
        for suffix, indexed, auxiliary in child_chains:
            sid = f"{trace_id}::sa:{agent_id}" + (f":{suffix}" if suffix else "")
            conversations.append(
                _conversation(
                    sid,
                    trace_id,
                    indexed,
                    "weka_subagent",
                    outer=outer,
                    auxiliary=auxiliary,
                )
            )
            child_ids.append(sid)
        preceding, join = group_key(outer, end, flat=False)
        branch_groups[("spawn", preceding, join)].append(
            (agent_id, start, end, child_ids)
        )
    for n, (suffix, indexed, auxiliary) in enumerate(chains[1:]):
        sid = f"{trace_id}::{suffix}"
        conversations.append(
            _conversation(sid, trace_id, indexed, "weka_flat", auxiliary=auxiliary)
        )
        start = indexed[0][1].t
        end = max(r.end_s for _, r in indexed)
        preceding, join = group_key(indexed[0][0], end, flat=True)
        branch_groups[("flatspawn", preceding, join)].append(
            (str(n), start, end, [sid])
        )
    branches = []
    for (kind, preceding, join), items in branch_groups.items():
        bid = f"{trace_id}:{kind}:{items[0][0]}"
        branch = Branch(
            bid,
            tuple(sid for item in items for sid in item[3]),
            preceding,
            join,
            min(item[1] for item in items),
        )
        branches.append(branch)
    predecessors = infer_predecessors(conversations)
    sources = set()
    for conversation in conversations:
        for turn in conversation.turns:
            if turn.source in sources:
                raise WorkloadValidationError(
                    f"duplicate original AgentX request: {turn.source}"
                )
            sources.add(turn.source)
            turn.predecessors = predecessors[turn.key]
    if sources != original_sources:
        raise WorkloadValidationError(
            f"AgentX source conservation failed in {trace_id}"
        )
    turns = [t for c in conversations for t in c.turns]
    return TraceGraph(
        trace_id,
        block_size,
        conversations,
        branches,
        raw_count,
        min(t.request.t for t in turns),
        max(t.request.t for t in turns),
        stats,
    )


def iter_trace_graphs(path: str | Path) -> Iterable[TraceGraph]:
    path = Path(path)
    paths = sorted(path.glob("*.json*")) if path.is_dir() else [path]
    seen = set()
    for file in paths:
        with file.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw = _loads(line)
                    graph = parse_trace_graph(raw)
                except (KeyError, TypeError, ValueError) as exc:
                    raise WorkloadValidationError(
                        f"invalid AgentX graph in {file}:{line_number}: {exc}"
                    ) from exc
                if graph.trace_id in seen:
                    raise WorkloadValidationError(
                        f"duplicate AgentX trace ID: {graph.trace_id}"
                    )
                seen.add(graph.trace_id)
                yield graph
