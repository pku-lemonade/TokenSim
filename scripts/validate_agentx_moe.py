#!/usr/bin/env python3
"""Validate complete AgentX routing/distribution runs and their GEMM baseline."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from TokenSim.config.config import ParallelConfig
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from validate_agentx_cpu import validate
from validate_agentx_gemm import _jsonl, _pressure, _profile_records, _request_map

BASELINE = ROOT / "results/agentx-dsv4-full/native-gemm/20261001-low-rank-projections"
BASELINE_EVIDENCE = ROOT / "data/reference/agentx-native-gemm/pipeline-validation.json"
RUNTIME_CHANGES = {
    "TokenSim/latency/operator_table.py",
    "TokenSim/llm/llm_engine.py",
    "TokenSim/moe/routing.py",
    "TokenSim/moe/stats.py",
    "TokenSim/operator_data/manifest.py",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    directory,
    baseline=BASELINE,
    baseline_evidence=BASELINE_EVIDENCE,
    *,
    hardware=("b200", "b300"),
):
    result = validate(directory, hardware=hardware, verify_reservations=True)
    result["scope"] = (
        "whole-step synthetic route statistics; fixed power_law_1.01 MXFP4 routed compute; "
        "native MegaMoE/shared expert, attention, mHC and MTP still pending"
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", "2f8e3d0^{commit}"], cwd=ROOT, text=True
    ).strip()
    previous = json.loads(baseline_evidence.read_text())["checkpoints"]
    context = HardwareContext.load(ROOT / "data")
    model = context.model("DeepSeek-V4-Pro")
    for sku, point in result["checkpoints"].items():
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
        for name, expected in old_manifest["source_sha256"].items():
            content = subprocess.check_output(
                ["git", "show", f"{commit}:{name}"], cwd=ROOT
            )
            assert hashlib.sha256(content).hexdigest() == expected, name
        old_sources, new_sources = (
            old_manifest["source_sha256"],
            new_manifest["source_sha256"],
        )
        changed = sorted(
            n
            for n in old_sources.keys() | new_sources.keys()
            if old_sources.get(n) != new_sources.get(n)
        )
        assert set(changed) == RUNTIME_CHANGES, changed
        for filename in (
            "kv_transfer_config.json",
            "native_replay/cache_configuration.json",
        ):
            assert json.loads((old_dir / filename).read_text()) == json.loads(
                (new_dir / filename).read_text()
            ), filename
        concurrency = new_manifest["point"]["concurrency"]
        name = f"agentx_c{concurrency}.json"
        assert digest(old_dir / name) == previous[sku]["result_sha256"]
        before = json.loads((old_dir / name).read_text())
        after = json.loads((new_dir / name).read_text())
        for field in ("parallel_config", "parallel_groups"):
            assert before[field] == after[field], field
        assert [w["cache"] for w in before["worker_cache_budgets"]] == [
            w["cache"] for w in after["worker_cache_budgets"]
        ]
        backend_description = after["latency_backends"][context.device(sku).device_id]
        assert backend_description["moe_cost_dtype"] == "mxfp4"
        assert backend_description["moe_cost_distribution_candidates"] == [
            "power_law_1.01"
        ]
        assert backend_description["moe_histogram_changes_latency"] is False
        assert after["moe_straggler_latency"] == 0
        assert (
            sum(after["operator_table_match_counts"]["moe"].values())
            == after["operator_step_count"]
        )
        component_time = sum(after["operator_component_seconds"].values())
        local_time = sum(
            after["dp_step_stats"]["local_compute_seconds_per_rank"].values()
        )
        assert abs(component_time - local_time) < 1e-6 * max(1, local_time)
        assert abs(
            after["operator_component_seconds"]["moe"] - after["moe_compute_latency"]
        ) < 1e-6 * max(1, after["moe_compute_latency"])
        loads = {int(k): v for k, v in after["moe_per_expert_load"].items()}
        assert sum(loads.values()) == after["moe_routed_token_count"]
        placement = after["moe_expert_placement"]["rank_to_experts"]
        assert {
            r: sum(loads.get(e, 0) for e in experts) for r, experts in placement.items()
        } == after["moe_per_rank_expert_load"]
        scheduled = sum(
            w["counts"]["scheduled_context_tokens"]
            + w["counts"]["scheduled_decode_tokens"]
            for w in after["scheduler_stats"]
        )
        dummy_width = (
            1 + new_manifest["defaults"]["speculative"]["num_speculative_tokens"]
        )
        dummy_tokens = (
            sum(after["dp_step_stats"]["dummy_batches_per_rank"].values()) * dummy_width
        )
        assert (scheduled + dummy_tokens) * model.moe.num_experts_per_tok == after[
            "moe_routed_token_count"
        ]
        assert len(loads) == model.moe.num_experts
        assert max(loads.values()) - min(loads.values()) <= len(
            after["worker_cache_budgets"]
        )
        assert all(w["loaded_waiting"] == 0 for w in point["final_workers"])
        assert all(w["running"] == w["waiting"] == 0 for w in point["final_workers"])
        records = _jsonl(new_dir / "native_replay/requests.jsonl")
        assert len({r["request_id"] for r in records}) == len(
            records
        ), "duplicate terminal request ID"
        for phase, expected in after["agentx_metrics"]["phase_request_counts"].items():
            observed = Counter(r["outcome"] for r in records if r["phase"] == phase)
            assert expected["sent"] == sum(observed.values()), (phase, "sent")
            assert all(
                outcome in {"successful", "cancelled", "error"} for outcome in observed
            )
            for outcome in ("successful", "cancelled", "error"):
                assert observed[outcome] == expected.get(outcome, 0), (phase, outcome)
        _request_map(_profile_records(new_dir), include_lane=True)
        package = context.operator_package(context.device(sku).device_id, "vllm")
        backend = OperatorTableLatencyBackend(
            device=context.device(sku),
            model=model,
            parallel_config=ParallelConfig(
                data_parallel_size=8, enable_expert_parallel=True
            ),
            package=package,
            fallback="table_first",
        )
        samples = []
        for tokens in (1, 8, 49, 96, 196, 384, 512, 8192, 16384):
            key = {
                "dtype": "mxfp4",
                "hidden_size": model.hidden_size,
                "inter_size": model.moe_intermediate_size,
                "top_k": model.moe.num_experts_per_tok,
                "num_experts": model.moe.num_experts,
                "tp_size": 1,
                "ep_size": 8,
                "num_tokens": tokens * 8,
            }
            estimate = backend.moe_query(key)
            samples.append(
                {
                    "local_tokens": tokens,
                    "global_tokens": key["num_tokens"],
                    "latency_us_one_layer": estimate.latency_us,
                    "match": estimate.match_type,
                    "distribution": estimate.workload_distribution,
                    "source_id": estimate.source_id,
                }
            )
        old_metrics, new_metrics = before["agentx_metrics"], after["agentx_metrics"]
        point["distribution_validation"] = {
            "route_scope": "one representative MoE layer per estimate, aggregated over workers; dummy batches included",
            "routing_histogram": "uniform synthetic statistics, rotating EP-interleaved remainder; not observed power-law routes",
            "cost_distribution": "power_law_1.01",
            "dtype": "mxfp4",
            "straggler_seconds": 0,
            "route_and_expert_placement_conservation": True,
            "one_moe_query_per_rank_estimate": True,
            "profile_component_time_conservation": True,
            "gpu_pool_and_running_limits": True,
            "phase_terminal_counts_and_unique_replay_identities": True,
            "sampled_queries": samples,
            "package_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in sorted(package.root.glob("*"))
                if p.is_file()
            },
        }
        point["comparison"] = {
            "baseline_commit": commit,
            "baseline_result_sha256": previous[sku]["result_sha256"],
            "changed_runtime_files": changed,
            "configuration_and_capacity_unchanged": True,
            "before": {
                "output_tps": old_metrics["output_token_throughput_tps"],
                "preemptions": before["preemption_count"],
                "expert_load_imbalance": before["moe_expert_load_imbalance_ratio"],
                "straggler_seconds": before["moe_straggler_latency"],
                "components": before["operator_component_seconds"],
            },
            "after": {
                "output_tps": new_metrics["output_token_throughput_tps"],
                "preemptions": after["preemption_count"],
                "expert_load_imbalance": after["moe_expert_load_imbalance_ratio"],
                "straggler_seconds": after["moe_straggler_latency"],
                "components": after["operator_component_seconds"],
            },
            "output_tps_change_percent": 100
            * (
                new_metrics["output_token_throughput_tps"]
                / old_metrics["output_token_throughput_tps"]
                - 1
            ),
            "diagnostic_limit": "closed-loop full windows; issue times and completed request sets change with execution cost",
            "before_pressure": _pressure(
                old_dir,
                before,
                _profile_records(old_dir),
                new_manifest["point"]["max_num_seqs"],
            ),
            "after_pressure": _pressure(
                new_dir,
                after,
                _profile_records(new_dir),
                new_manifest["point"]["max_num_seqs"],
            ),
        }
    result["readonly_reference_source_sha256"] = {
        name: digest(Path("/home/zhuohang/Serving/vllm-note") / name)
        for name in ("vllm/models/deepseek_v4/nvidia/model.py",)
    }
    return result


def plot_comparison(directory, baseline, result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    skus = tuple(result["checkpoints"])
    figure, axes = plt.subplots(
        2, len(skus), figsize=(5 * len(skus), 6), squeeze=False, constrained_layout=True
    )
    for column, sku in enumerate(skus):
        point = result["checkpoints"][sku]
        manifest = json.loads((directory / sku / "run_manifest.json").read_text())
        concurrency = manifest["point"]["concurrency"]
        before = json.loads(
            (baseline / sku / f"agentx_c{concurrency}.json").read_text()
        )
        after = json.loads(
            (directory / sku / f"agentx_c{concurrency}.json").read_text()
        )
        ranks = np.arange(len(after["moe_per_rank_expert_load"]))
        axis = axes[0, column]
        for full, offset, color, label in (
            (before, -0.18, "#8b98a5", "Before"),
            (after, 0.18, "#2677bb", "After"),
        ):
            values = [full["moe_per_rank_expert_load"][str(r)] for r in ranks]
            mean = sum(values) / len(values)
            axis.bar(
                ranks + offset,
                [v / mean for v in values],
                width=0.36,
                color=color,
                label=label,
            )
        axis.set_title(sku.upper())
        axis.set_xticks(ranks)
        axis.set_xlabel("EP rank")
        axis.set_ylabel("Synthetic routed load / rank mean")
        axis.axhline(1, color="#555555", linestyle="--", linewidth=0.8)
        axis.legend(fontsize=8)
        reference = json.loads(
            (
                ROOT
                / "data/reference"
                / f"{sku}-c{concurrency}-dep8-simplecpu/key_point.json"
            ).read_text()
        )["metrics"]
        reference_tps = (
            reference["output_tput_per_gpu"] * manifest["defaults"]["num_gpus"]
        )
        axis = axes[1, column]
        values = [
            point["comparison"]["before"]["output_tps"],
            point["output_tps"],
            reference_tps,
        ]
        axis.bar(
            ("Before", "After", "Reference"),
            values,
            color=("#8b98a5", "#2677bb", "#79a16a"),
        )
        for index, value in enumerate(values):
            axis.text(
                index, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=9
            )
        axis.set_ylim(0, max(values) * 1.15)
        axis.set_ylabel("Output tokens/s (all 8 GPUs)")
        for row in range(2):
            axes[row, column].grid(axis="y", alpha=0.2)
            axes[row, column].set_axisbelow(True)
    figure.suptitle(
        "Full 393-trace pool, 3600-second closed-loop measurement\nRoute statistics are synthetic; latency uses power_law_1.01"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--baseline-evidence", type=Path, default=BASELINE_EVIDENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    parser.add_argument(
        "--hardware", choices=("b200", "b300"), nargs="+", default=("b200", "b300")
    )
    args = parser.parse_args()
    result = audit(
        args.directory, args.baseline, args.baseline_evidence, hardware=args.hardware
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        plot_comparison(args.directory, args.baseline, result, args.plot)
    print(
        json.dumps(
            {sku: p["comparison"] for sku, p in result["checkpoints"].items()}, indent=2
        )
    )
