#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BLOG_CACHE_RATES = {
    "b200": {"hbm": 0.73, "dram": 0.20},
    "b300": {"hbm": 0.91, "dram": 0.0136},
}
BLOG_OFFLOAD_IMPROVEMENT = {
    "output_throughput_pct": 81.7,
    "mean_e2e_reduction_pct": 46.6,
}


def percent_delta(actual: float, reference: float) -> float:
    return (actual / reference - 1.0) * 100.0 if reference else 0.0


def cache_rates(result: dict[str, Any], mode: str) -> tuple[float, float]:
    if mode == "dram":
        denominator = result.get("mooncake_cache_query_tokens", 0)
    else:
        denominator = (
            result.get("reuse_hit_blocks", 0) + result.get("reuse_miss_blocks", 0)
        ) * result.get("kv_cache_block_size", 64)
    if not denominator:
        return 0.0, 0.0
    if mode == "dram":
        hbm_tokens = result.get("mooncake_local_gpu_hit_tokens", 0)
        dram_tokens = result.get("mooncake_memory_hit_tokens", 0)
    else:
        hbm_tokens = result.get("reuse_hit_blocks", 0) * result.get(
            "kv_cache_block_size", 64
        )
        dram_tokens = 0
    return hbm_tokens / denominator, dram_tokens / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    args = parser.parse_args()
    root = Path(args.result_root)
    references = {
        row["hardware"]: row
        for row in json.loads(
            (root / "reference" / "inferencex_key_points.json").read_text()
        )
    }

    rows = []
    raw_results: dict[tuple[str, str], dict[str, Any]] = {}
    for hardware, concurrency in (("b200", 196), ("b300", 384)):
        reference = references[hardware]
        reference_metrics = reference["metrics"]
        for mode in ("hbm", "dram"):
            result_path = root / hardware / mode / f"agentx_c{concurrency}.json"
            result = json.loads(result_path.read_text())
            raw_results[(hardware, mode)] = result
            metrics = result["agentx_metrics"]
            hbm_rate, dram_rate = cache_rates(result, mode)
            output_per_gpu = metrics["output_token_throughput_per_gpu_tps"]
            input_per_gpu = (
                metrics["input_tokens"]
                / metrics["profile_duration_s"]
                / reference["num_gpus"]
            )
            ttft_p90 = metrics["ttft_s"]["p90"]
            tpot_p90 = metrics["tpot_s"]["p90"]
            mean_e2e = metrics["e2e_s"]["mean"]
            rows.append(
                {
                    "hardware": hardware,
                    "concurrency": concurrency,
                    "cache_mode": mode,
                    "request_count": metrics["request_count"],
                    "profile_duration_s": metrics["profile_duration_s"],
                    "output_tput_per_gpu": output_per_gpu,
                    "reference_output_tput_per_gpu": (
                        reference_metrics["output_tput_per_gpu"]
                        if mode == "dram"
                        else ""
                    ),
                    "output_tput_delta_pct": (
                        percent_delta(
                            output_per_gpu,
                            reference_metrics["output_tput_per_gpu"],
                        )
                        if mode == "dram"
                        else ""
                    ),
                    "input_tput_per_gpu": input_per_gpu,
                    "reference_input_tput_per_gpu": (
                        reference_metrics["input_tput_per_gpu"]
                        if mode == "dram"
                        else ""
                    ),
                    "input_tput_delta_pct": (
                        percent_delta(
                            input_per_gpu,
                            reference_metrics["input_tput_per_gpu"],
                        )
                        if mode == "dram"
                        else ""
                    ),
                    "p90_ttft_s": ttft_p90,
                    "reference_p90_ttft_s": (
                        reference_metrics["p90_ttft"] if mode == "dram" else ""
                    ),
                    "p90_ttft_delta_pct": (
                        percent_delta(ttft_p90, reference_metrics["p90_ttft"])
                        if mode == "dram"
                        else ""
                    ),
                    "p90_tpot_s": tpot_p90,
                    "reference_p90_itl_s": (
                        reference_metrics["p90_itl"] if mode == "dram" else ""
                    ),
                    "p90_tpot_vs_itl_delta_pct": (
                        percent_delta(tpot_p90, reference_metrics["p90_itl"])
                        if mode == "dram"
                        else ""
                    ),
                    "mean_e2e_s": mean_e2e,
                    "reference_mean_e2e_s": (
                        reference_metrics["mean_e2e"] if mode == "dram" else ""
                    ),
                    "mean_e2e_delta_pct": (
                        percent_delta(mean_e2e, reference_metrics["mean_e2e"])
                        if mode == "dram"
                        else ""
                    ),
                    "hbm_hit_rate": hbm_rate,
                    "blog_hbm_hit_rate": (
                        BLOG_CACHE_RATES[hardware]["hbm"] if mode == "dram" else ""
                    ),
                    "hbm_hit_rate_delta_pp": (
                        (hbm_rate - BLOG_CACHE_RATES[hardware]["hbm"]) * 100
                        if mode == "dram"
                        else ""
                    ),
                    "dram_hit_rate": dram_rate,
                    "blog_dram_hit_rate": (
                        BLOG_CACHE_RATES[hardware]["dram"] if mode == "dram" else ""
                    ),
                    "dram_hit_rate_delta_pp": (
                        (dram_rate - BLOG_CACHE_RATES[hardware]["dram"]) * 100
                        if mode == "dram"
                        else ""
                    ),
                    "kv_cache_capacity_tokens": result.get(
                        "cluster", {}
                    ).get(
                        "kv_cache_capacity_tokens_total",
                        result.get("kv_cache_capacity_tokens_total", 0),
                    ),
                    "reference_kv_cache_capacity_tokens": reference_metrics[
                        "kv_cache_pool_tokens"
                    ],
                }
            )

    improvements = []
    for hardware in ("b200", "b300"):
        hbm_metrics = raw_results[(hardware, "hbm")]["agentx_metrics"]
        dram_metrics = raw_results[(hardware, "dram")]["agentx_metrics"]
        throughput_gain = percent_delta(
            dram_metrics["output_token_throughput_tps"],
            hbm_metrics["output_token_throughput_tps"],
        )
        e2e_reduction = (
            1.0
            - dram_metrics["e2e_s"]["mean"] / hbm_metrics["e2e_s"]["mean"]
        ) * 100.0
        improvements.append(
            {
                "hardware": hardware,
                "output_throughput_gain_pct": throughput_gain,
                "blog_output_throughput_gain_pct": BLOG_OFFLOAD_IMPROVEMENT[
                    "output_throughput_pct"
                ],
                "throughput_gain_gap_pp": throughput_gain
                - BLOG_OFFLOAD_IMPROVEMENT["output_throughput_pct"],
                "mean_e2e_reduction_pct": e2e_reduction,
                "blog_mean_e2e_reduction_pct": BLOG_OFFLOAD_IMPROVEMENT[
                    "mean_e2e_reduction_pct"
                ],
                "mean_e2e_reduction_gap_pp": e2e_reduction
                - BLOG_OFFLOAD_IMPROVEMENT["mean_e2e_reduction_pct"],
            }
        )

    comparison = {
        "methodology": {
            "dataset_trace_count": 393,
            "profile_duration_s": 1800,
            "system_idle_gap_cap_s": 10,
            "simulator_model": "DeepSeek-V4-Proxy",
            "official_submission_compatible": False,
            "cache_rate_denominator": (
                "Mooncake connector prompt-query tokens for DRAM cases; "
                "completed-request prefix blocks for HBM-only cases"
            ),
            "note": (
                "Simulation follows the current public AIPerf MVP timing defaults; "
                "the article describes an earlier one-hour/per-stream-five-minute-cap run."
            ),
        },
        "rows": rows,
        "offload_improvements": improvements,
    }
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    with (root / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(root / "comparison.json")
    print(root / "comparison.csv")


if __name__ == "__main__":
    main()
