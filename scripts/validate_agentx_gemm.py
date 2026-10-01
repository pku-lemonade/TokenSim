#!/usr/bin/env python3
"""Audit native projection shapes and full-pipeline component accounting."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.deepseek_v4_gemm import attention_projections
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.config.config import ParallelConfig
from validate_agentx_cpu import validate


BASELINE_DIRECTORY = (
    ROOT / "results/agentx-dsv4-full/simple-cpu/20261001-loaded-kv-waiting"
)
BASELINE_EVIDENCE = (
    ROOT / "data/reference/agentx-loaded-kv-admission/pipeline-validation.json"
)
CHANGED_RUNTIME_FILES = {
    "TokenSim/latency/deepseek_v4_gemm.py",
    "TokenSim/latency/operator_table.py",
    "TokenSim/llm/llm_engine.py",
    "TokenSim/operator_data/analytical/gpu.py",
    "TokenSim/operator_data/analytical/groq.py",
    "TokenSim/operator_data/manifest.py",
    "TokenSim/operator_data/schema.py",
    "TokenSim/operator_data/workload.py",
}
SOURCE_REQUEST_FIELDS = (
    "trace_id",
    "conversation_id",
    "outer_index",
    "inner_index",
    "pass_index",
)


def _jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def _profile_records(directory):
    return [
        r
        for r in _jsonl(directory / "native_replay/requests.jsonl")
        if r["phase"] == "profiling"
    ]


def _request_map(records, *, include_lane):
    # The correlation UUID also includes lane, so matching just the original
    # source request cannot establish identical router or cache identity.
    fields = SOURCE_REQUEST_FIELDS + (("lane",) if include_lane else ())
    result = {tuple(r[k] for k in fields): r for r in records}
    assert len(result) == len(records), "duplicate profiling request identity"
    return result


def _pressure(directory, full, records, max_num_seqs):
    windows = _jsonl(directory / "native_replay/cache_windows.jsonl")
    profile = [w for w in windows if w["phase"] == "profiling"]
    assert profile, "missing profiling cache windows"
    final = windows[-1]
    ranks = []
    for budget in full["worker_cache_budgets"]:
        rank = budget["worker_id"]
        capacity = budget["cache"]["num_gpu_blocks"]
        samples = [w["workers"][rank] for w in profile]
        for window in windows:
            worker = window["workers"][rank]
            assert worker["used_slots"] + worker["free_slots"] == capacity
            assert 1 <= worker["used_slots"] <= capacity
            assert worker["running"] <= max_num_seqs, "running sequence limit"
        worker = final["workers"][rank]
        assert worker["loaded_waiting"] == 0
        requests = [r for r in records if r["worker_id"] == rank]
        ranks.append(
            {
                "rank": rank,
                "sent": len(requests),
                "repeat_admissions_all_outcomes": sum(
                    max(0, r["prefill_admission_count"] - 1) for r in requests
                ),
                "completed_recomputations_all_outcomes": sum(
                    r["recomputation_count"] for r in requests
                ),
                "reloaded_cpu_tokens_all_outcomes": sum(
                    r["reloaded_cpu_tokens"] for r in requests
                ),
                "capacity_slots": capacity,
                "max_sample_used_fraction": max(w["used_slots"] for w in samples)
                / capacity,
                "mean_sample_used_fraction": sum(w["used_slots"] for w in samples)
                / len(samples)
                / capacity,
                "near_full_sample_count": sum(
                    w["used_slots"] >= 0.95 * capacity for w in samples
                ),
                "mean_sample_waiting": sum(
                    w["waiting"] + w["loading"] + w["loaded_waiting"] for w in samples
                )
                / len(samples),
                "loaded_capacity_gates": worker["scheduler_stats"][
                    "loaded_kv_capacity_gate"
                ],
                "async_reservation_gates": worker["scheduler_stats"][
                    "async_load_reservation_gate"
                ],
            }
        )
    return {
        "sample_count": len(profile),
        "sampling": "profiling windows; sample means, not time integrals",
        "per_rank": ranks,
    }


def _compare(directory, baseline, baseline_evidence, baseline_commit, sku, full):
    old_dir, new_dir = baseline / sku, directory / sku
    old_manifest = json.loads((old_dir / "run_manifest.json").read_text())
    new_manifest = json.loads((new_dir / "run_manifest.json").read_text())
    for field in (
        "point",
        "defaults",
        "backend",
        "dataset_sha256",
        "transfer_utilization_assumption",
    ):
        assert old_manifest[field] == new_manifest[field], field
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{baseline_commit}^{{commit}}"],
        cwd=ROOT,
        text=True,
    ).strip()
    for name, expected in old_manifest["source_sha256"].items():
        original = subprocess.check_output(
            ["git", "show", f"{commit}:{name}"], cwd=ROOT
        )
        assert hashlib.sha256(original).hexdigest() == expected, name
    old_sources, new_sources = (
        old_manifest["source_sha256"],
        new_manifest["source_sha256"],
    )
    changed = sorted(
        name
        for name in old_sources.keys() | new_sources.keys()
        if old_sources.get(name) != new_sources.get(name)
    )
    assert set(changed) <= CHANGED_RUNTIME_FILES, changed
    for name in ("kv_transfer_config.json", "native_replay/cache_configuration.json"):
        assert json.loads((old_dir / name).read_text()) == json.loads(
            (new_dir / name).read_text()
        ), name
    c = new_manifest["point"]["concurrency"]
    before_path = old_dir / f"agentx_c{c}.json"
    previous_evidence = json.loads(baseline_evidence.read_text())["checkpoints"][sku]
    assert (
        hashlib.sha256(before_path.read_bytes()).hexdigest()
        == previous_evidence["result_sha256"]
    )
    before = json.loads(before_path.read_text())
    assert json.loads((old_dir / "run_status.json").read_text())["exit_code"] == 0
    for field in ("parallel_config", "parallel_groups"):
        assert before[field] == full[field], field
    assert [w["cache"] for w in before["worker_cache_budgets"]] == [
        w["cache"] for w in full["worker_cache_budgets"]
    ], "cache budgets"
    old_records, new_records = _profile_records(old_dir), _profile_records(new_dir)
    old_successful = [r for r in old_records if r["outcome"] == "successful"]
    assert len(old_successful) == before["agentx_metrics"]["request_count"]
    for field in ("input_tokens", "output_tokens"):
        assert sum(r[field] for r in old_successful) == before["agentx_metrics"][field]
    old_source, new_source = _request_map(
        old_records, include_lane=False
    ), _request_map(new_records, include_lane=False)
    source_common = old_source.keys() & new_source.keys()
    assert all(
        (old_source[k]["input_tokens"], old_source[k]["requested_output_tokens"])
        == (new_source[k]["input_tokens"], new_source[k]["requested_output_tokens"])
        for k in source_common
    ), "source lengths"
    old_replay, new_replay = _request_map(old_records, include_lane=True), _request_map(
        new_records, include_lane=True
    )
    replay_common = old_replay.keys() & new_replay.keys()
    assert all(
        old_replay[k]["dp_rank"] == new_replay[k]["dp_rank"] for k in replay_common
    ), "same-session route drift"
    old_metrics, new_metrics = before["agentx_metrics"], full["agentx_metrics"]
    return {
        "baseline_commit": commit,
        "baseline_result_sha256": previous_evidence["result_sha256"],
        "baseline_records_sha256": hashlib.sha256(
            (old_dir / "native_replay/requests.jsonl").read_bytes()
        ).hexdigest(),
        "changed_runtime_files": changed,
        "configuration_and_capacity_unchanged": True,
        "output_tps_before": old_metrics["output_token_throughput_tps"],
        "output_tps_change_percent": 100
        * (
            new_metrics["output_token_throughput_tps"]
            / old_metrics["output_token_throughput_tps"]
            - 1
        ),
        "preemptions_before": before["preemption_count"],
        "successful_cpu_reload_tokens_before": old_metrics["reloaded_cpu_tokens"],
        "successful_cpu_reload_tokens_after": new_metrics["reloaded_cpu_tokens"],
        "source_requests_shared": len(source_common),
        "source_requests_routed_to_different_ranks": sum(
            old_source[k]["dp_rank"] != new_source[k]["dp_rank"] for k in source_common
        ),
        "replay_identity_fields": list(SOURCE_REQUEST_FIELDS + ("lane",)),
        "replay_requests_shared": len(replay_common),
        "same_replay_identity_route_mismatches": 0,
        "diagnostic_limit": "closed-loop request issue times and trace/lane assignment change; this is not a fixed-arrival causal experiment",
        "before_pressure": _pressure(
            old_dir, before, old_records, new_manifest["point"]["max_num_seqs"]
        ),
        "after_pressure": _pressure(
            new_dir, full, new_records, new_manifest["point"]["max_num_seqs"]
        ),
    }


def audit(
    directory,
    *,
    hardware=("b200", "b300"),
    baseline=BASELINE_DIRECTORY,
    baseline_evidence=BASELINE_EVIDENCE,
    baseline_commit="18eba15",
):
    result = validate(directory, hardware=hardware, verify_reservations=True)
    context = HardwareContext.load(ROOT / "data")
    model = context.model("DeepSeek-V4-Pro")
    result["scope"] = (
        "native low-rank GEMMs and per-operator dtype; remaining native modules/MTP pending"
    )
    for sku, point in result["checkpoints"].items():
        c = 196 if sku == "b200" else 384
        full = json.loads((directory / sku / f"agentx_c{c}.json").read_text())
        components = full["operator_component_seconds"]
        local_time = sum(
            full["dp_step_stats"]["local_compute_seconds_per_rank"].values()
        )
        # Unlike the old report, these counters exclude warmup and count
        # every real/dummy estimate of the synchronized profiling passes.
        assert abs(sum(components.values()) - local_time) < 1e-6 * max(1, local_time)
        probes = []
        b = OperatorTableLatencyBackend(
            device=context.device(sku),
            model=model,
            parallel_config=ParallelConfig(),
            package=context.operator_package(context.device(sku).device_id, "vllm"),
            fallback="table_first",
        )
        for tokens in (1, 49, 96, 512, 8192, 16384):
            for p in attention_projections(model, 1):
                key = p.key(tokens)
                estimate = b._query(p.table, key)
                probes.append(
                    {
                        "projection": p.name,
                        "table": p.table,
                        "key": key,
                        "latency_us": estimate.latency_us,
                        "match": estimate.match_type,
                        "source_id": estimate.source_id,
                    }
                )
        point["native_gemm"] = {
            "projections": [
                {"name": p.name, "table": p.table, **p.key(1)}
                for p in attention_projections(model, 1)
            ],
            "probes": probes,
            "profile_component_seconds": components,
            "profile_component_fraction": {
                k: v / sum(components.values()) for k, v in components.items()
            },
            "component_sum_equals_local_compute": True,
            "warmup_excluded": True,
            "grouped_einsum_data": "no matched measurements; analytical geometry with one launch",
        }
        point["comparison"] = _compare(
            directory, baseline, baseline_evidence, baseline_commit, sku, full
        )
    source = Path("/home/zhuohang/Serving/vllm-note")
    result["source_sha256"] = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (
            "vllm/models/deepseek_v4/attention.py",
            "vllm/models/deepseek_v4/nvidia/ops/o_proj.py",
            "vllm/models/deepseek_v4/nvidia/model.py",
            "vllm/v1/worker/gpu/model_runner.py",
            "vllm/v1/worker/gpu/input_batch.py",
        )
    }
    return result


def plot_comparison(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    skus = tuple(result["checkpoints"])
    figure, axes = plt.subplots(
        2, len(skus), figsize=(5 * len(skus), 6), constrained_layout=True, squeeze=False
    )
    ranks = np.arange(8)
    for col, sku in enumerate(skus):
        comparison = result["checkpoints"][sku]["comparison"]
        for row, (field, scale, label) in enumerate(
            (
                ("max_sample_used_fraction", 100, "Maximum sampled GPU pool use (%)"),
                (
                    "repeat_admissions_all_outcomes",
                    1,
                    "Repeat admissions (all profile outcomes)",
                ),
            )
        ):
            axis = axes[row, col]
            for name, offset, color, legend in (
                ("before_pressure", -0.18, "#8b98a5", "Before: generic projections"),
                ("after_pressure", 0.18, "#2677bb", "After: native projections"),
            ):
                values = [p[field] * scale for p in comparison[name]["per_rank"]]
                axis.bar(ranks + offset, values, width=0.36, color=color, label=legend)
            axis.set_title(sku.upper())
            axis.set_xticks(ranks)
            axis.set_xlabel("DP rank")
            axis.set_ylabel(label)
            axis.set_ylim(0, 110 if row == 0 else 80)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            if row == 0:
                axis.legend(fontsize=8)
    figure.suptitle(
        "Full 393-trace pool, 3600-second closed-loop measurement\nPool maxima use 60 profiling samples; request sets differ"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=BASELINE_DIRECTORY)
    parser.add_argument("--baseline-evidence", type=Path, default=BASELINE_EVIDENCE)
    parser.add_argument("--baseline-commit", default="18eba15")
    parser.add_argument("--plot", type=Path)
    parser.add_argument(
        "--hardware", choices=("b200", "b300"), nargs="+", default=("b200", "b300")
    )
    args = parser.parse_args()
    result = audit(
        args.directory,
        hardware=args.hardware,
        baseline=args.baseline,
        baseline_evidence=args.baseline_evidence,
        baseline_commit=args.baseline_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        plot_comparison(result, args.plot)
    print(
        json.dumps(
            {
                sku: {
                    "output_tps": p["output_tps"],
                    "components": p["native_gemm"]["profile_component_fraction"],
                }
                for sku, p in result["checkpoints"].items()
            },
            indent=2,
        )
    )
