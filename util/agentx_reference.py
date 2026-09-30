"""Audit AgentX reference artifacts without running AIPerf or the simulator.

Request metrics follow AIPerf's exported units and InferenceX's successful
profiling-record aggregation. Server endpoints must be explicitly associated
with the same engine namespace before their series can be deduplicated.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STATS = ("mean", "p50", "p75", "p90", "p95", "p99", "std")


def load_experiments(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported AgentX matrix schema")
    defaults = raw.get("defaults", {})
    experiments = []
    ids = set()
    for entry in raw["experiments"]:
        exp = {**defaults, **entry}
        if exp["id"] in ids:
            raise ValueError(f"duplicate experiment ID: {exp['id']}")
        ids.add(exp["id"])
        if not isinstance(exp["num_gpus"], int) or exp["num_gpus"] <= 0:
            raise ValueError("num_gpus must be a positive integer")
        experiments.append(exp)
    return experiments


def file_identity(path: Path, *, public_source: str | None = None) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "public_source": public_source,
    }


def distribution(values: Iterable[float]) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return {"count": 0, **dict.fromkeys(STATS)}
    if not np.isfinite(a).all():
        raise ValueError("nonfinite reference metric")
    return {
        "count": len(a),
        "mean": float(a.mean()),
        **{f"p{p}": float(np.percentile(a, p)) for p in (50, 75, 90, 95, 99)},
        "std": float(a.std()),
    }


def metric(record: dict[str, Any], name: str, *, seconds: bool = False) -> float | None:
    item = record.get("metrics", {}).get(name)
    if item is None:
        return None
    value = item.get("value") if isinstance(item, dict) else item
    if value is None:
        return None
    if seconds:
        unit = item.get("unit") if isinstance(item, dict) else None
        scales = {"s": 1, "sec": 1, "seconds": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
        if unit not in scales:
            raise ValueError(f"unknown time unit for {name}: {unit}")
        value *= scales[unit]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {name}")
    return value


def read_profile(path: Path) -> list[dict[str, Any]]:
    # Keep only fields needed for this audit, not HTTP timing/chunk details.
    names = (
        "input_sequence_length",
        "output_sequence_length",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "request_latency",
        "time_to_first_token",
        "inter_token_latency",
        "osl_mismatch_diff_pct",
    )
    records = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                records.append(
                    {
                        "metadata": row["metadata"],
                        "error": row.get("error"),
                        "metrics": {k: row.get("metrics", {}).get(k) for k in names},
                    }
                )
    return records


def successful_profile(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in records
        if r["metadata"].get("benchmark_phase", "profiling") == "profiling"
        and not r.get("error")
        and not r["metadata"].get("was_cancelled", False)
    ]


def request_metrics(records: list[dict[str, Any]], num_gpus: int) -> dict[str, Any]:
    rows = successful_profile(records)
    if not rows:
        raise ValueError("no successful profiling records")
    # Preserve integer nanoseconds until subtraction (epoch floats lose precision).
    start = min(r["metadata"]["request_start_ns"] for r in rows)
    end = max(r["metadata"]["request_end_ns"] for r in rows)
    duration = (end - start) / 1e9
    if duration <= 0 or num_gpus <= 0:
        raise ValueError("positive observation duration and GPU count required")
    values: dict[str, list[float]] = defaultdict(list)
    exported_itl_errors = []
    for r in rows:
        isl, osl = metric(r, "input_sequence_length"), metric(
            r, "output_sequence_length"
        )
        ttft = metric(r, "time_to_first_token", seconds=True)
        e2e = metric(r, "request_latency", seconds=True)
        if any(v is None for v in (isl, osl, ttft, e2e)):
            raise ValueError("successful record missing input/output/TTFT/latency")
        values["isl"].append(isl)
        values["osl"].append(osl)
        if ttft > 0:
            values["ttft_s"].append(ttft)
        if e2e > 0:
            values["e2e_s"].append(e2e)
        if osl > 1:
            itl = (e2e - ttft) / (osl - 1)
            if itl > 0:
                values["itl_s"].append(itl)
            exported = metric(r, "inter_token_latency", seconds=True)
            if exported is not None:
                exported_itl_errors.append(abs(itl - exported))
    inputs, outputs = sum(values["isl"]), sum(values["osl"])
    return {
        "request_count": len(rows),
        "start_ns": start,
        "end_ns": end,
        "profile_duration_s": duration,
        "input_tokens": int(inputs),
        "output_tokens": int(outputs),
        "input_tput_per_gpu": inputs / duration / num_gpus,
        "output_tput_per_gpu": outputs / duration / num_gpus,
        "input_tput_tps": inputs / duration,
        "output_tput_tps": outputs / duration,
        "itl_export_max_abs_error_s": max(exported_itl_errors, default=None),
        **{
            k: distribution(values[k])
            for k in ("isl", "osl", "ttft_s", "e2e_s", "itl_s")
        },
    }


def benchmark_accounting(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        match = re.search(
            r"Phase (warmup|profiling) \(\1\) complete \| completed=([\d,]+), "
            r"cancelled=([\d,]+), errors=([\d,]+).*elapsed=([\d.]+)s",
            line,
        )
        if match:
            phase, completed, cancelled, errors, elapsed = match.groups()
            result.setdefault(phase, {}).update(
                completed=int(completed.replace(",", "")),
                cancelled=int(cancelled.replace(",", "")),
                wire_errors=int(errors.replace(",", "")),
                elapsed_s=float(elapsed),
            )
        match = re.search(
            r"Phase (warmup|profiling).*?sending complete \| sent=([\d,]+)", line
        )
        if not match:
            match = re.search(
                r"Phase (warmup|profiling) timed out.*Stats: sent=(\d+)", line
            )
        if match:
            result.setdefault(match[1], {})["sent"] = int(match[2].replace(",", ""))
    return result


def accounting(records: list[dict[str, Any]], wire: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for phase in ("warmup", "profiling"):
        rows = [
            r
            for r in records
            if r["metadata"].get("benchmark_phase", "profiling") == phase
        ]
        cancelled = sum(bool(r["metadata"].get("was_cancelled")) for r in rows)
        errors = sum(
            bool(r.get("error")) and not r["metadata"].get("was_cancelled", False)
            for r in rows
        )
        stats = {
            "records": len(rows),
            "successful_records": len(rows) - errors - cancelled,
            "error_records": errors,
            "cancelled_records": cancelled,
            "trace_count": len({r["metadata"]["source_trace_id"] for r in rows}),
            "wire": wire.get(phase, {}),
        }
        if "cancelled" in stats["wire"]:
            stats["cancelled_without_record"] = stats["wire"]["cancelled"] - cancelled
        if "sent" in stats["wire"] and "completed" in stats["wire"]:
            stats["wire_conservation"] = (
                stats["wire"]["sent"]
                == stats["wire"]["completed"] + stats["wire"]["cancelled"]
            )
        result[phase] = stats
    return result


def trace_lengths(
    path: Path,
) -> tuple[dict[tuple, int], dict[str, list[int]], list[str]]:
    lengths, legacy, order = {}, {}, []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            trace = json.loads(line)
            tid = trace["id"]
            if tid in legacy:
                raise ValueError(f"duplicate trace ID: {tid}")
            order.append(tid)
            legacy[tid] = []
            for outer, req in enumerate(trace["requests"]):
                if "requests" in req:
                    for inner, child in enumerate(req["requests"]):
                        lengths[(tid, outer, inner)] = int(child["out"])
                else:
                    lengths[(tid, outer, None)] = int(req["out"])
                    if req["type"] in ("n", "s"):
                        legacy[tid].append(int(req["out"]))
    return lengths, legacy, order


def output_length_audit(
    records: list[dict[str, Any]],
    lengths: dict[tuple, int],
    legacy: dict[str, list[int]],
) -> dict[str, Any]:
    expected, actual, legacy_expected = [], [], []
    missing = []
    mismatches = []
    for r in successful_profile(records):
        m = r["metadata"]
        key = (m["source_trace_id"], m["source_outer_idx"], m.get("source_inner_idx"))
        if key not in lengths:
            missing.append(key)
            continue
        target = max(1, lengths[key])  # AIPerf Turn.max_tokens >= 1.
        observed = metric(r, "output_sequence_length")
        expected.append(target)
        actual.append(observed)
        if observed != target:
            mismatches.append({"source": key, "actual": observed, "expected": target})
        # Reproduce the historical aggregate's lookup solely to diagnose it.
        turns = legacy.get(m["conversation_id"].split("::", 1)[0], [])
        index = m["turn_index"]
        if 0 <= index < len(turns):
            legacy_expected.append(turns[index])
    return {
        "mapping": "source_trace_id + source_outer_idx + source_inner_idx; zero OSL becomes one",
        "expected": distribution(expected),
        "actual": distribution(actual),
        "historical_turn_index_lookup": distribution(legacy_expected),
        "missing_sources": missing,
        "mismatches": mismatches,
        "nonzero_exported_mismatch_count": sum(
            metric(r, "osl_mismatch_diff_pct") not in (None, 0)
            for r in successful_profile(records)
        ),
    }


def deduplicate_series(
    series: list[dict[str, Any]],
    endpoint_groups: dict[str, str],
    preferred_endpoint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in series:
        endpoint = row["endpoint_url"]
        labels = row.get("labels", {})
        if "engine" not in labels:
            raise ValueError("cannot deduplicate a rank metric without engine identity")
        # Unlisted endpoints belong to independent namespaces, even for engine=0.
        namespace = endpoint_groups.get(endpoint, endpoint)
        groups[(namespace, tuple(sorted(labels.items())))].append(row)
    chosen, audit = [], []
    for (namespace, labels), rows in groups.items():
        endpoints = [r["endpoint_url"] for r in rows]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("duplicate metric identity within a single endpoint")
        selected = min(
            rows,
            key=lambda r: (r["endpoint_url"] != preferred_endpoint, r["endpoint_url"]),
        )
        chosen.append(selected)
        audit.append(
            {
                "namespace": namespace,
                "labels": dict(labels),
                "selected_endpoint": selected["endpoint_url"],
                "endpoints": endpoints,
                "stats_by_endpoint": {r["endpoint_url"]: r["stats"] for r in rows},
            }
        )
    return chosen, audit


SERVER_NAMES = (
    "prompt_tokens",
    "generation_tokens",
    "prefix_cache_hits",
    "prefix_cache_queries",
    "external_prefix_cache_hits",
    "external_prefix_cache_queries",
    "prompt_tokens_by_source",
    "kv_cache_usage_perc",
    "num_requests_running",
    "num_requests_waiting",
    "num_preemptions",
)


def server_metrics(
    path: Path, exp: dict[str, Any], timeseries_path: Path
) -> dict[str, Any]:
    # The original exports are 1.3/1.4 GB. Materialize one metric at a time,
    # stream selected per-rank gauge slices to disk, and discard each metric.
    import ijson

    counters, per_rank, audits, missing = {}, defaultdict(dict), {}, set(SERVER_NAMES)
    with path.open("rb") as stream, timeseries_path.open("w") as output:
        for name, data in ijson.kvitems(stream, "metrics", use_float=True):
            short = name.removeprefix("vllm:")
            if name != "vllm:" + short or short not in SERVER_NAMES:
                continue
            missing.discard(short)
            selected, audit = deduplicate_series(
                data["series"],
                exp["metrics_endpoint_groups"],
                exp["preferred_metrics_endpoint"],
            )
            audits[short] = audit
            totals = [r["stats"].get("total") for r in selected]
            if all(v is not None for v in totals):
                counters[short] = sum(totals)
            for r in selected:
                rank = r["labels"]["engine"]
                source = r["labels"].get("source")
                key = f"{short}:{source}" if source is not None else short
                per_rank[rank][key] = r["stats"]
                if short in (
                    "kv_cache_usage_perc",
                    "num_requests_running",
                    "num_requests_waiting",
                ):
                    for point in r.get("timeslices", []):
                        output.write(
                            json.dumps(
                                {
                                    "rank": rank,
                                    "metric": short,
                                    "endpoint": r["endpoint_url"],
                                    **point,
                                }
                            )
                            + "\n"
                        )
    if not all(name in counters for name in ("prompt_tokens", "generation_tokens")):
        raise ValueError("server metrics missing prompt/generation counters")
    return {
        "counters": counters,
        "per_rank": dict(per_rank),
        "deduplication": audits,
        "missing_metrics": sorted(missing),
        "namespace_rule": "explicit endpoint aliases plus all series labels; direct engine endpoint preferred",
    }


def server_summary(path: Path) -> dict[str, Any]:
    import ijson

    with path.open("rb") as stream:
        return next(ijson.items(stream, "summary", use_float=True))


def router_metrics(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    pattern = re.compile(
        r"CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:([^']+)' "
        r"mapped to worker: (\S+)@(\d+)"
    )
    assignments, loads = defaultdict(set), Counter()
    with path.open() as stream:
        for line in stream:
            match = pattern.search(ANSI.sub("", line))
            if match:
                correlation, endpoint, rank = match.groups()
                assignments[correlation].add(int(rank))
                loads[rank] += 1
    if not loads:
        raise ValueError("router log contains no recognized session assignments")
    observed = [r for r in records if r["metadata"]["x_correlation_id"] in assignments]
    children = [r for r in observed if r["metadata"].get("parent_correlation_id")]
    cross_rank = {k: sorted(v) for k, v in assignments.items() if len(v) != 1}
    return {
        "request_count": sum(loads.values()),
        "session_count": len(assignments),
        "rank_request_counts": dict(loads),
        "cross_rank_sessions": cross_rank,
        "record_mapping_count": len(observed),
        "record_count": len(records),
        "record_mapping_coverage": len(observed) / len(records) if records else None,
        "child_records": len(children),
        "child_id_equal_parent_count": sum(
            r["metadata"]["x_correlation_id"] == r["metadata"]["parent_correlation_id"]
            for r in children
        ),
        "session_to_rank": {
            k: next(iter(v)) for k, v in assignments.items() if len(v) == 1
        },
        "rank_identity": "worker URL @suffix; router index is position in the worker list",
    }


def server_log(
    path: Path, output_path: Path, phase_ranges: dict[str, Any] | None = None
) -> dict[str, Any]:
    scheduler = re.compile(r"Reqs Running: (\d+), Waiting: (\d+)")
    spec = re.compile(
        r"Mean acceptance length: ([\d.]+).*Per-position acceptance rate: ([\d., ]+), Avg"
    )
    capacity, weights, kv_bytes = {}, {}, {}
    rows, scheduler_count, spec_count, scheduler_lines = [], 0, 0, 0
    phase_ranges = phase_ranges or {}
    profile_start = phase_ranges.get("profiling", {}).get("start_ns")
    year = (
        datetime.fromtimestamp(profile_start / 1e9, timezone.utc).year
        if profile_start
        else None
    )
    with path.open() as stream, output_path.open("w") as output:
        for original in stream:
            line = ANSI.sub("", original)
            scheduler_lines += "Reqs Running:" in line
            rank = re.search(r"(?:EngineCore|Worker)_DP(\d+)", line)
            if rank:
                for expression, target in (
                    (r"GPU KV cache size: ([\d,]+) tokens", capacity),
                    (
                        r"(?:Loading model weights took|Model loading took) ([\d.]+) GiB",
                        weights,
                    ),
                    (r"Available KV cache memory: ([\d.]+) GiB", kv_bytes),
                ):
                    match = re.search(expression, line)
                    if match:
                        target[rank[1]] = float(match[1].replace(",", ""))
            stamp = re.search(r"\b\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b", line)
            timestamp = stamp[0] if stamp else None
            epoch_ns = (
                int(
                    datetime.strptime(f"{year}-{timestamp}", "%Y-%m-%d %H:%M:%S")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1e9
                )
                if timestamp and year
                else None
            )
            phase = next(
                (
                    name
                    for name, bounds in phase_ranges.items()
                    if epoch_ns is not None
                    and bounds["start_ns"] <= epoch_ns <= bounds["end_ns"]
                ),
                None,
            )
            match = scheduler.search(line)
            if match:
                # Rust frontend inserts Deferred between Waiting and GPU KV,
                # and omits hit metrics when no query was observed. Parse
                # fields independently so those samples are never dropped.
                def fraction(label: str):
                    found = re.search(rf"{label}: ([\d.]+)%", line)
                    return float(found[1]) / 100 if found else None

                deferred = re.search(r"Deferred: (\d+) reqs", line)
                row = {
                    "kind": "scheduler",
                    "timestamp": timestamp,
                    "epoch_ns": epoch_ns,
                    "phase": phase,
                    "running": int(match[1]),
                    "waiting": int(match[2]),
                    "deferred": int(deferred[1]) if deferred else None,
                    "gpu_kv_fraction": fraction("GPU KV cache used"),
                    "gpu_prefix_hit_fraction": fraction("Prefix cache hit rate"),
                    "external_hit_fraction_of_gpu_misses": fraction(
                        "External prefix cache hit rate"
                    ),
                }
                rows.append(row)
                scheduler_count += 1
                output.write(json.dumps(row) + "\n")
            match = spec.search(line)
            if match:
                output.write(
                    json.dumps(
                        {
                            "kind": "speculative",
                            "timestamp": timestamp,
                            "epoch_ns": epoch_ns,
                            "phase": phase,
                            "acceptance_length": float(match[1]),
                            "per_position_acceptance": [
                                float(x) for x in match[2].split(",")
                            ],
                        }
                    )
                    + "\n"
                )
                spec_count += 1
    return {
        "scheduler_samples": scheduler_count,
        "unparsed_scheduler_samples": scheduler_lines - scheduler_count,
        "speculative_samples": spec_count,
        "logged_equivalent_kv_tokens_per_rank": capacity,
        "logged_equivalent_kv_tokens_total": int(sum(capacity.values())),
        "weight_gib_per_rank": weights,
        "available_kv_gib_per_rank": kv_bytes,
        "running_all_log_samples": distribution(r["running"] for r in rows),
        "running_profiling_log_samples": distribution(
            r["running"] for r in rows if r["phase"] == "profiling"
        ),
        "scope": "all timestamps retained; profiling summary uses exported phase boundaries; Rust frontend counts are aggregate",
        "timestamp_assumption": "log timestamps interpreted as UTC, matching the saved export epoch and wall-clock boundaries",
    }


def sampling_audit(path: Path, order: list[str], seed: int) -> dict[str, Any]:
    lanes = {}
    with path.open() as stream:
        for line in stream:
            match = re.search(
                r"lane=(\d+)\s+sample_time=\s*(\d+)%.*trace_id=(\S+)", line
            )
            if match:
                lanes[int(match[1])] = {
                    "trace_id": match[3],
                    "sample_time_pct_rounded": int(match[2]),
                }
    shuffled = list(order)
    random.Random(seed).shuffle(shuffled)
    return {
        "observed_initial_lanes": lanes,
        "sequential_order_matches": sum(
            row["trace_id"] == order[lane % len(order)] for lane, row in lanes.items()
        ),
        "legacy_shuffled_order_matches": sum(
            row["trace_id"] == shuffled[lane % len(order)]
            for lane, row in lanes.items()
        ),
        "workload_seed": seed,
        "reference_recycle": "shared sequential sampler, next draw when any tree completes",
        "legacy_recycle": "(lane + play_id * concurrency) modulo a seed-shuffled pool",
    }


def bmk_comparison(
    metrics: dict[str, Any], bmk: dict[str, Any], *, tolerance_pct: float = 0.5
) -> list[dict[str, Any]]:
    request = bmk["request_metrics"]
    comparisons = []
    for name, reference in (
        ("profile_duration_s", request["throughput"]["duration_seconds"]),
        ("input_tput_tps", request["throughput"]["input"]["tokens_per_second"]),
        ("output_tput_tps", request["throughput"]["output"]["tokens_per_second"]),
    ):
        comparisons.append((name, metrics[name], reference))
    for name, reference in (
        ("ttft_s", request["latency"]["ttft"]),
        ("itl_s", request["latency"]["itl"]),
        ("e2e_s", request["latency"]["e2el"]),
        ("isl", request["tokens"]["input"]),
        ("osl", request["tokens"]["output_actual"]),
    ):
        for stat in STATS:
            if stat in reference:
                comparisons.append(
                    (f"{name}.{stat}", metrics[name][stat], reference[stat])
                )
    result = []
    for name, actual, reference in comparisons:
        delta = (
            (actual / reference - 1) * 100
            if reference
            else (0 if actual == 0 else None)
        )
        # Published values are rounded to five decimals (especially ITL std).
        passed = abs(actual - reference) <= 0.0000051 or (
            delta is not None and abs(delta) <= tolerance_pct
        )
        result.append(
            {
                "metric": name,
                "actual": actual,
                "bmk": reference,
                "delta_pct": delta,
                "passed": passed,
            }
        )
    return result
