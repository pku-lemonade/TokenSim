#!/usr/bin/env python3
"""Audit every original request, reference identity, dependency and initial lane."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TokenSim.workload.agentx_support.parser import iter_trace_graphs
from TokenSim.workload.agentx_support.trajectory import (
    SequentialTracePool,
    initial_snapshots,
    primer_schedule,
)
from util.agentx_reference import file_identity, load_experiments


def validate(dataset: Path, artifacts: Path, matrix: Path, output: Path) -> dict:
    started = time.perf_counter()
    graphs = []
    totals, widths = Counter(), Counter()
    sources = {}
    for graph in iter_trace_graphs(dataset):
        graphs.append(graph)
        totals["traces"] += 1
        totals["requests"] += graph.request_count
        totals["conversations"] += len(graph.conversations)
        totals["branches"] += len(graph.branches)
        for conversation in graph.conversations:
            totals["auxiliary_conversations"] += conversation.is_auxiliary
            for turn in conversation.turns:
                sources[
                    (
                        turn.source.trace_id,
                        turn.source.outer_index,
                        turn.source.inner_index,
                    )
                ] = (turn.key.conversation_id, turn.key.turn_index, turn.source_kind)
                if turn.predecessors:
                    totals["gated_turns"] += 1
                    widths[len(turn.predecessors)] += 1
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "kind": "native-request-graph-audit",
        "simulation_performed": False,
        "dataset": file_identity(dataset),
        "matrix": file_identity(matrix),
        "totals": dict(totals),
        "predecessor_widths": dict(sorted(widths.items())),
        "experiments": [],
    }
    failures = []
    for exp in load_experiments(matrix):
        reference_root = artifacts / exp["reference"]["artifact_subdir"] / "agentic"
        profile = reference_root / "aiperf_artifacts/profile_export.jsonl"
        records, matches = 0, 0
        identity_mismatches = []
        with profile.open() as stream:
            for line in stream:
                record = json.loads(line)
                metadata = record["metadata"]
                key = (
                    metadata["source_trace_id"],
                    metadata["source_outer_idx"],
                    metadata.get("source_inner_idx"),
                )
                observed = sources.get(key)
                expected = (
                    metadata["conversation_id"],
                    metadata["turn_index"],
                    metadata["source_kind"],
                )
                records += 1
                if observed == expected:
                    matches += 1
                elif len(identity_mismatches) < 20:
                    identity_mismatches.append(
                        {"source": key, "computed": observed, "reference": expected}
                    )
        log_path = reference_root / "benchmark.log"
        log = log_path.read_text()
        barriers = re.findall(
            r"Replay interval barriers active: (\d+) requests, (\d+) gated turns, join-widths=(\{[^\n]+?\})",
            log,
        )
        if not barriers:
            raise ValueError("reference benchmark log has no dependency accounting")
        barrier_matches = all(
            int(requests) == totals["requests"]
            and int(gated) == totals["gated_turns"]
            and ast.literal_eval(histogram) == dict(widths)
            for requests, gated, histogram in barriers
        )
        snapshots = initial_snapshots(
            SequentialTracePool(graphs),
            exp["concurrency"],
            workload_seed=exp["workload_seed"],
        )
        logged_lanes = {}
        for match in re.finditer(
            r"lane=(\d+)\s+sample_time=\s*(\d+)%\s+root_next=\s*(\d+)/(\d+)\s+"
            r"\([^\n]+?\)\s+live=(\d+) ready=(\d+)\s+trace_id=(\S+)",
            log,
        ):
            lane, percentage, root_index, root_count, live, ready, tid = match.groups()
            logged_lanes[int(lane)] = (
                tid,
                int(percentage),
                int(root_index),
                int(root_count),
                int(live),
                int(ready),
            )
        lane_mismatches = []
        for snapshot in snapshots:
            duration = snapshot.graph.time_end_s - snapshot.graph.time_start_s
            pct = (
                (snapshot.t_star_ms / 1000 - snapshot.graph.time_start_s)
                / duration
                * 100
                if duration
                else 0
            )
            observed = (
                snapshot.graph.trace_id,
                round(pct),
                snapshot.root_next_turn_index,
                len(snapshot.graph.root.turns),
                len(snapshot.states),
                sum(not state.waiting_on_children for state in snapshot.states),
            )
            if observed != logged_lanes.get(snapshot.lane_index):
                lane_mismatches.append(
                    {
                        "lane": snapshot.lane_index,
                        "computed": observed,
                        "reference": logged_lanes.get(snapshot.lane_index),
                    }
                )
        primers = primer_schedule(snapshots)
        primer_match = re.search(
            r"10 additional requests per lane after (\d+) mandatory snapshot primers",
            log,
        )
        if not primer_match:
            raise ValueError("reference benchmark log has no primer accounting")
        checks = {
            "complete_trace_pool": totals["traces"] == exp["trace_count"],
            "original_request_conservation": len(sources) == totals["requests"],
            "reference_request_identity": records == matches,
            "dependency_counts_and_widths": barrier_matches,
            "all_initial_lanes_match": not lane_mismatches
            and len(logged_lanes) == exp["concurrency"],
            "mandatory_primer_count": len(primers) == int(primer_match[1]),
        }
        result = {
            "experiment_id": exp["id"],
            "reference_id": exp["reference"]["id"],
            "checks": checks,
            "profile_records": records,
            "matching_records": matches,
            "identity_mismatches": identity_mismatches,
            "lane_mismatches": lane_mismatches,
            "initial_lanes": len(snapshots),
            "mandatory_primers": len(primers),
            "primer_spread_s": max((offset for offset, _, _ in primers), default=0),
            "input_files": [file_identity(profile), file_identity(log_path)],
        }
        report["experiments"].append(result)
        if not all(checks.values()):
            failures.append(exp["id"])
        print(
            f"{exp['id']}: identity={matches}/{records}, lanes={len(snapshots)}, "
            f"primers={len(primers)}, passed={all(checks.values())}",
            flush=True,
        )
    report["wall_time_s"] = time.perf_counter() - started
    report["failed_experiments"] = failures
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset/agentx/traces.jsonl")
    )
    parser.add_argument("--reference-artifacts", type=Path, required=True)
    parser.add_argument(
        "--matrix", type=Path, default=Path("data/experiments/agentx.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.dataset, args.reference_artifacts, args.matrix, args.output)
    if report["failed_experiments"]:
        raise SystemExit(
            f"native graph reference validation failed: {report['failed_experiments']}"
        )


if __name__ == "__main__":
    main()
