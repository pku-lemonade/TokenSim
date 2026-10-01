#!/usr/bin/env python3
"""Reconcile full CPU runs against request logs, pool state and reference metrics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.agentx_accuracy import error_assessment


def validate(
    directory,
    *,
    hardware=("b200", "b300"),
    verify_reservations=False,
    verify_sources=True,
):
    points = {}
    for sku, concurrency in (("b200", 196), ("b300", 384)):
        if sku not in hardware:
            continue
        point = directory / sku
        assert json.loads((point / "run_status.json").read_text())["exit_code"] == 0
        manifest = json.loads((point / "run_manifest.json").read_text())
        if verify_sources:
            for name, expected in manifest["source_sha256"].items():
                assert (
                    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
                ), name
        assert (
            manifest["dataset_sha256"]
            == hashlib.sha256(
                (ROOT / "dataset/agentx/traces.jsonl").read_bytes()
            ).hexdigest()
        )
        path = point / f"agentx_c{concurrency}.json"
        result = json.loads(path.read_text())
        metrics = result["agentx_metrics"]
        records = [
            json.loads(line)
            for line in (point / "native_replay/requests.jsonl")
            .read_text()
            .splitlines()
        ]
        successful = [
            r
            for r in records
            if r["phase"] == "profiling" and r["outcome"] == "successful"
        ]
        assert len(successful) == metrics["request_count"]
        if metrics.get("cache_source_observation") == "first-admission":
            assert all(
                r["prefill_source_observed"] and r["prefill_admission_count"] >= 1
                for r in successful
            )
            for field, record_field in (
                ("readmission_count", "prefill_admission_count"),
                ("reloaded_gpu_tokens", "reloaded_gpu_tokens"),
                ("reloaded_cpu_tokens", "reloaded_cpu_tokens"),
                ("recomputed_tokens", "recomputed_tokens"),
                ("latest_gpu_hit_tokens", "latest_gpu_hit_tokens"),
                ("latest_cpu_hit_tokens", "latest_cpu_hit_tokens"),
            ):
                total = sum(
                    r[record_field] - int(field == "readmission_count")
                    for r in successful
                )
                assert metrics[field] == total, field
        assert all(
            r["output_tokens"] == r["requested_output_tokens"] for r in successful
        )
        for field, record_field in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("gpu_cached_input_tokens", "gpu_hit_tokens"),
            ("cpu_cached_input_tokens", "cpu_hit_tokens"),
            ("computed_input_tokens", "computed_input_tokens"),
        ):
            assert metrics[field] == sum(r[record_field] for r in successful), field
        assert metrics["input_tokens"] == (
            metrics["gpu_cached_input_tokens"]
            + metrics["cpu_cached_input_tokens"]
            + metrics["computed_input_tokens"]
        )
        cancelled = [r for r in records if r["outcome"] == "cancelled"]
        assert all(r["cancellation_acknowledged"] for r in cancelled)
        windows = [
            json.loads(line)
            for line in (point / "native_replay/cache_windows.jsonl")
            .read_text()
            .splitlines()
        ]
        final = windows[-1]
        if verify_reservations:
            assert all(
                w["inflight_prefills"] == 0
                and w["inflight_prefill_reserved_slots"] == 0
                for w in final["workers"]
            )
            assert all(
                w["inflight_prefill_reserved_slots"] >= 0
                for window in windows
                for w in window["workers"]
            )
        assert metrics["trace_count"] == 393 and metrics["issue_window_s"] == 3600
        assert metrics["cancel_unreturned"] == 0
        assert all(
            w["used_slots"] == 1
            and w["dma_pinned_slots"] == 0
            and w["loading"] == 0
            and w["external_cache"]["storage"]["pinned_pages"] == 0
            and w["external_cache"]["write_jobs"] == 0
            for w in final["workers"]
        )
        dp = final["dp_step"]
        compute = sum(dp["local_compute_seconds_per_rank"].values())
        wait = sum(dp["synchronization_wait_seconds_per_rank"].values())
        assert abs(compute + wait - 8 * dp["elapsed_step_seconds"]) < 1e-6
        ref_path = (
            ROOT
            / "data/reference"
            / (f"{sku}-c{concurrency}-dep8-simplecpu/key_point.json")
        )
        ref = json.loads(ref_path.read_text())["metrics"]
        actual = {
            "output_tput_per_gpu": metrics["output_token_throughput_per_gpu_tps"],
            "p90_ttft": metrics["ttft_s"]["p90"],
            "p90_itl": metrics["tpot_s"]["p90"],
            "gpu_cache_hit_rate": metrics["gpu_cache_hit_rate"],
            "cpu_cache_hit_rate": metrics["cpu_cache_hit_rate"],
        }
        points[sku] = {
            "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "phase_counts": metrics["phase_request_counts"],
            "warmup_seconds": metrics["warmup_duration_s"],
            "phase_elapsed_seconds": final["simulation_time"]
            - metrics["warmup_duration_s"],
            "observation_seconds": metrics["profile_duration_s"],
            "output_tps": metrics["output_token_throughput_tps"],
            "gpu_hit_percent": metrics["gpu_cache_hit_rate"] * 100,
            "cpu_hit_percent": metrics["cpu_cache_hit_rate"] * 100,
            "cache_source_observation": metrics.get(
                "cache_source_observation", "latest-admission"
            ),
            "ttft_p90": metrics["ttft_s"]["p90"],
            "tpot_p90": metrics["tpot_s"]["p90"],
            "e2e_mean": metrics["e2e_s"]["mean"],
            "sync_wait_fraction": wait / (compute + wait),
            "mean_successful_input_tokens": metrics["input_tokens"] / len(successful),
            "mean_successful_output_tokens": metrics["output_tokens"] / len(successful),
            "preemptions": result["preemption_count"],
            "recomputed_tokens": result["recomputed_tokens"],
            "external_hit_fraction_of_gpu_misses": metrics["cpu_cache_hit_rate"]
            / (1 - metrics["gpu_cache_hit_rate"]),
            "diagnostic_accuracy": {
                k: error_assessment(v, ref.get(k)) for k, v in actual.items()
            },
            "final_workers": final["workers"],
            "per_rank": [
                {
                    "rank": rank,
                    "successful": sum(r["dp_rank"] == rank for r in successful),
                    "cancelled": sum(r["dp_rank"] == rank for r in cancelled),
                    "input_tokens": sum(
                        r["input_tokens"] for r in successful if r["dp_rank"] == rank
                    ),
                    "gpu_hit_tokens": sum(
                        r["gpu_hit_tokens"] for r in successful if r["dp_rank"] == rank
                    ),
                    "cpu_hit_tokens": sum(
                        r["cpu_hit_tokens"] for r in successful if r["dp_rank"] == rank
                    ),
                    "max_used_gpu_slots": max(
                        w["workers"][rank]["used_slots"]
                        for w in windows
                        if w["phase"] in ("profiling", "complete")
                    ),
                    "full_input_gate": final["workers"][rank]["scheduler_stats"][
                        "full_input_gate"
                    ],
                }
                for rank in range(8)
            ],
            "checks": "counts/output/input decomposition/cancellation/pool/DP conservation passed",
        }
    return {
        "scope": "native SimpleCPU asynchronous pipeline; generic operators and one-token decode",
        "formal_model_accuracy_accepted": False,
        "trace_pool": 393,
        "issue_window_seconds": 3600,
        "link_counters": "cumulative warmup+profiling; connector result counters reset after warmup",
        "checkpoints": points,
        "inflight_reservations_verified": verify_reservations,
    }


def plot_pressure(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row, (sku, concurrency) in enumerate((("b200", 196), ("b300", 384))):
        point = directory / sku
        metrics = json.loads((point / f"agentx_c{concurrency}.json").read_text())[
            "agentx_metrics"
        ]
        start = metrics["warmup_duration_s"]
        windows = [
            json.loads(line)
            for line in (point / "native_replay/cache_windows.jsonl")
            .read_text()
            .splitlines()
        ]
        windows = [w for w in windows if w["phase"] in ("profiling", "complete")]
        times = [(w["simulation_time"] - start) / 60 for w in windows]
        for rank in range(8):
            usage = [
                100
                * w["workers"][rank]["used_slots"]
                / (w["workers"][rank]["used_slots"] + w["workers"][rank]["free_slots"])
                for w in windows
            ]
            axes[row, 0].plot(times, usage, label=str(rank), linewidth=0.8)
        axes[row, 0].set(
            title=f"{sku.upper()} HBM slots", ylabel="used / total (%)", ylim=(0, 102)
        )
        requests = [
            json.loads(line)
            for line in (point / "native_replay/requests.jsonl")
            .read_text()
            .splitlines()
        ]
        bins = np.zeros((61, 3))
        for req in requests:
            if req["phase"] != "profiling" or req["outcome"] != "successful":
                continue
            bucket = min(60, int((req["end_s"] - start) // 60))
            bins[bucket] += [
                req["input_tokens"],
                req["gpu_hit_tokens"],
                req["cpu_hit_tokens"],
            ]
        rates = (
            np.divide(
                bins[:, 1:],
                bins[:, :1],
                out=np.full((61, 2), np.nan),
                where=bins[:, :1] != 0,
            )
            * 100
        )
        for column, name in enumerate(("GPU", "CPU")):
            axes[row, 1].plot(np.arange(61) + 0.5, rates[:, column], label=name)
        axes[row, 1].set(
            title="Completed input source by minute",
            ylabel="input tokens (%)",
            ylim=(0, 102),
        )
        waits = [
            sum(
                worker["external_cache"]["read"]["queue_wait_seconds"]
                for worker in w["workers"]
            )
            for w in windows
        ]
        axes[row, 2].plot(times, [v - waits[0] for v in waits])
        axes[row, 2].set(
            title="Read queue wait across 8 ranks", ylabel="cumulative seconds"
        )
        axes[row, 2].set_ylim(0, max(0.001, (waits[-1] - waits[0]) * 1.05))
        for ax in axes[row]:
            ax.set(xlabel="profiling minutes", xlim=(0, 61))
            ax.grid(alpha=0.2)
        axes[row, 0].legend(title="DP rank", ncol=4, fontsize=7)
        axes[row, 1].legend()
    figure.suptitle("Native SimpleCPU pipeline; generic costs / one-token decode")
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--verify-reservations", action="store_true")
    parser.add_argument(
        "--hardware", choices=("b200", "b300"), nargs="+", default=("b200", "b300")
    )
    args = parser.parse_args()
    result = validate(
        args.directory,
        hardware=args.hardware,
        verify_reservations=args.verify_reservations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        plot_pressure(args.directory, args.plot)
    print(
        json.dumps(
            {
                sku: {
                    k: p[k]
                    for k in ("output_tps", "gpu_hit_percent", "cpu_hit_percent")
                }
                for sku, p in result["checkpoints"].items()
            },
            indent=2,
        )
    )
