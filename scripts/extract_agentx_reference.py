#!/usr/bin/env python3
"""Rebuild reference points from the entire saved AgentX profiling runs."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.agentx_reference import (
    accounting,
    benchmark_accounting,
    bmk_comparison,
    file_identity,
    load_experiments,
    output_length_audit,
    read_profile,
    request_metrics,
    router_metrics,
    sampling_audit,
    server_log,
    server_metrics,
    server_summary,
    trace_lengths,
)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def extract(
    artifact_root: Path,
    dataset: Path,
    matrix: Path,
    output: Path,
    source_files: list[Path],
    licenses: list[Path],
) -> dict:
    started = time.perf_counter()
    lengths, legacy, order = trace_lengths(dataset)
    experiments = load_experiments(matrix)
    expected_trace_counts = {e["trace_count"] for e in experiments}
    if expected_trace_counts != {len(order)}:
        raise ValueError(
            f"matrix trace counts {expected_trace_counts} differ from dataset {len(order)}"
        )
    # Never overwrite an earlier reference audit or the untracked trace pool.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "kind": "reference-audit",
        "dataset": file_identity(
            dataset,
            public_source="https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126",
        ),
        "trace_count": len(order),
        "original_request_count": len(lengths),
        "matrix": file_identity(matrix),
        "sources": [],
        "licenses": [],
        "experiments": [],
    }
    for source in source_files:
        # Save the inspected source as well as its digest so /tmp is not a
        # permanent dependency of the evidence. It is never imported/executed.
        destination = output / "sources" / source.name
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise ValueError(f"duplicate source basename: {source.name}")
        shutil.copyfile(source, destination)
        manifest["sources"].append(
            file_identity(source) | {"saved_as": str(destination.relative_to(output))}
        )
    for license in licenses:
        destination = (
            output / "licenses" / f"{len(manifest['licenses'])}-{license.name}"
        )
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(license, destination)
        manifest["licenses"].append(
            file_identity(license) | {"saved_as": str(destination.relative_to(output))}
        )
    points, failures = [], []
    for exp in experiments:
        exp_start = time.perf_counter()
        reference = exp["reference"]
        root = artifact_root / reference["artifact_subdir"]
        agentic = root / "agentic"
        artifacts = agentic / "aiperf_artifacts"
        bmks = list((root / "bmk").glob("*.json"))
        if len(bmks) != 1:
            raise ValueError(f"expected exactly one bmk in {root / 'bmk'}")
        bmk = json.loads(bmks[0].read_text())
        aggregate = json.loads((artifacts / "profile_export_aiperf.json").read_text())
        phase = aggregate["input_config"]["phases"][0]
        if (
            bmk["conc"] != exp["concurrency"]
            or phase["concurrency"] != exp["concurrency"]
            or phase["duration"] != exp["profiling_window_s"]
            or phase["grace_period"] != exp["grace_period_s"]
        ):
            raise ValueError(f"reference window/concurrency mismatch: {exp['id']}")
        command = (agentic / "vllm_command.txt").read_text()
        option_names = {
            "tp": "tensor-parallel-size",
            "dp": "data-parallel-size",
            "block_size": "block-size",
            "max_num_seqs": "max-num-seqs",
            "max_num_batched_tokens": "max-num-batched-tokens",
            "prefill_schedule_interval": "prefill-schedule-interval",
            "long_prefill_token_threshold": "long-prefill-token-threshold",
        }
        runtime_options = {}
        for key, flag in option_names.items():
            found = re.search(rf"--{flag}\s+(\d+)\b", command)
            if found is None:
                raise ValueError(f"missing --{flag} in reference launch command")
            runtime_options[key] = int(found[1])
            expected = exp["parallelism"].get(key) if key in ("tp", "dp") else exp[key]
            if runtime_options[key] != expected:
                raise ValueError(f"reference {key} mismatch: {exp['id']}")
        if runtime_options["tp"] * runtime_options["dp"] * bmk["pp"] != exp["num_gpus"]:
            raise ValueError(
                "reference GPU count disagrees with explicit TP/DP/PP configuration"
            )
        if (
            bmk["ep"] != exp["parallelism"]["ep"]
            or "--enable-expert-parallel" not in command
        ):
            raise ValueError("reference EP configuration mismatch")
        target = output / exp["id"]
        target.mkdir()
        records = read_profile(artifacts / "profile_export.jsonl")
        metrics = request_metrics(records, exp["num_gpus"])
        checks = bmk_comparison(metrics, bmk)
        counts = accounting(records, benchmark_accounting(agentic / "benchmark.log"))
        osl = output_length_audit(records, lengths, legacy)
        router = router_metrics(agentic / "router.log", records)
        sampling = sampling_audit(
            agentic / "benchmark.log", order, exp["workload_seed"]
        )
        summary = server_summary(artifacts / "server_metrics_export.json")
        log = server_log(
            agentic / "server.log",
            target / "server_log_timeseries.jsonl",
            summary["phase_time_ranges"],
        )
        server = server_metrics(
            artifacts / "server_metrics_export.json",
            exp,
            target / "rank_timeseries.jsonl",
        )
        server["export_summary"] = summary
        # Keep usage totals separate from server counters, which also include
        # unfinished/error requests and different scrape boundaries.
        usage = {}
        for phase_name in ("warmup", "profiling"):
            rows = [
                r
                for r in records
                if r["metadata"].get("benchmark_phase") == phase_name
                and not r.get("error")
                and not r["metadata"].get("was_cancelled")
            ]
            usage[phase_name] = {
                name: sum(
                    r["metrics"][name]["value"] for r in rows if r["metrics"].get(name)
                )
                for name in ("usage_prompt_tokens", "usage_completion_tokens")
            }
        server["successful_record_usage"] = usage
        server["server_minus_successful_profiling_usage"] = {
            "prompt_tokens": server["counters"]["prompt_tokens"]
            - usage["profiling"]["usage_prompt_tokens"],
            "generation_tokens": server["counters"]["generation_tokens"]
            - usage["profiling"]["usage_completion_tokens"],
        }
        cache = bmk["server_metrics"]["cache"]
        by_source = {}
        for source in ("local_cache_hit", "external_kv_transfer", "local_compute"):
            by_source[source] = sum(
                rank[f"prompt_tokens_by_source:{source}"]["total"]
                for rank in server["per_rank"].values()
            )
        prompt_source_total = sum(by_source.values())
        recomputed_cache = {
            "gpu_cache_hit_rate": by_source["local_cache_hit"] / prompt_source_total,
            "cpu_cache_hit_rate": by_source["external_kv_transfer"]
            / prompt_source_total,
        }
        server["prompt_tokens_by_source"] = by_source
        server["recomputed_cache_rates"] = recomputed_cache
        for key, actual in recomputed_cache.items():
            checks.append(
                {
                    "metric": key,
                    "actual": actual,
                    "bmk": cache[key],
                    "passed": abs(actual - cache[key]) <= 0.0000051
                    or abs(actual / cache[key] - 1) <= 0.005,
                }
            )
        point = {
            "schema_version": 1,
            "experiment_id": exp["id"],
            "reference_id": reference["id"],
            "hardware": exp["hardware"],
            "concurrency": exp["concurrency"],
            "num_gpus": exp["num_gpus"],
            "parallelism": exp["parallelism"],
            "cache_backend": exp["cache_backend"],
            "model": exp["model"],
            "metrics": metrics
            | {
                "p90_ttft": metrics["ttft_s"]["p90"],
                "p90_itl": metrics["itl_s"]["p90"],
                "mean_e2e": metrics["e2e_s"]["mean"],
                "gpu_cache_hit_rate": cache["gpu_cache_hit_rate"],
                "cpu_cache_hit_rate": cache["cpu_cache_hit_rate"],
                "kv_cache_pool_tokens": log["logged_equivalent_kv_tokens_total"],
            },
            "kv_cache_pool_tokens_semantics": "logged equivalent tokens; not usable capacity or simulator input",
            "cache_rate_source": "bmk prompt_tokens_by_source; denominator is all prompt tokens in that export",
            "window": {
                "issue_s": phase["duration"],
                "grace_s": phase["grace_period"],
                "metric_observation_s": metrics["profile_duration_s"],
            },
            "reference_runtime_options": runtime_options,
        }
        checks += [
            {
                "metric": "request_count",
                "passed": metrics["request_count"] == bmk["num_requests_successful"],
            },
            {
                "metric": "source_mapping",
                "passed": not osl["missing_sources"] and not osl["mismatches"],
            },
            {
                "metric": "logged_equivalent_kv_tokens",
                "passed": point["metrics"]["kv_cache_pool_tokens"]
                == bmk["kv_cache_pool_tokens"],
            },
            {"metric": "router_stability", "passed": not router["cross_rank_sessions"]},
            {
                "metric": "router_record_coverage",
                "passed": router["record_mapping_count"] == len(records),
            },
            {
                "metric": "wire_accounting",
                "passed": all(
                    counts[p].get("wire_conservation") is True for p in counts
                ),
            },
            {
                "metric": "initial_lane_order",
                "passed": sampling["sequential_order_matches"] == exp["concurrency"],
            },
            {
                "metric": "historical_expected_osl_bug_reproduced",
                "passed": abs(
                    osl["historical_turn_index_lookup"]["mean"]
                    - bmk["request_metrics"]["tokens"]["output_expected"]["mean"]
                )
                <= 0.0000051,
            },
            {
                "metric": "engine_count",
                "passed": len(log["logged_equivalent_kv_tokens_per_rank"])
                == exp["parallelism"]["dp"],
            },
            {
                "metric": "all_scheduler_log_samples_parsed",
                "passed": log["unparsed_scheduler_samples"] == 0,
            },
        ]
        if not all(c["passed"] for c in checks):
            failures.append(exp["id"])
        for name, value in (
            ("key_point", point),
            ("bmk_validation", checks),
            ("request_accounting", counts),
            ("output_length_audit", osl),
            ("router", router),
            ("sampling", sampling),
            ("server_metrics", server),
            ("server_log", log),
        ):
            write_json(target / f"{name}.json", value)
        points.append(point)
        shutil.copyfile(bmks[0], target / "original_bmk.json")
        inputs = [
            bmks[0],
            artifacts / "profile_export.jsonl",
            artifacts / "profile_export_aiperf.json",
            artifacts / "server_metrics_export.json",
            agentic / "benchmark.log",
            agentic / "router.log",
            agentic / "server.log",
            agentic / "benchmark_command.txt",
            agentic / "vllm_command.txt",
        ]
        manifest["experiments"].append(
            {
                "config": exp,
                "inputs": [file_identity(p) for p in inputs],
                "checks": len(checks),
                "passed": all(c["passed"] for c in checks),
                "wall_time_s": time.perf_counter() - exp_start,
            }
        )
        print(
            f"{exp['id']}: records={len(records)}, successful={metrics['request_count']}, "
            f"checks={len(checks)}, passed={all(c['passed'] for c in checks)}",
            flush=True,
        )
    write_json(output / "inferencex_key_points.json", points)
    manifest["wall_time_s"] = time.perf_counter() - started
    manifest["failed_experiments"] = failures
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--matrix", type=Path, default=Path("data/experiments/agentx.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/agentx-dsv4-full/reference")
    )
    parser.add_argument("--source-file", type=Path, action="append", default=[])
    parser.add_argument("--license", type=Path, action="append", default=[])
    args = parser.parse_args()
    manifest = extract(
        args.artifact_root,
        args.dataset,
        args.matrix,
        args.output,
        args.source_file,
        args.license,
    )
    if manifest["failed_experiments"]:
        raise SystemExit(f"reference checks failed: {manifest['failed_experiments']}")


if __name__ == "__main__":
    main()
