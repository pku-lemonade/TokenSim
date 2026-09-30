from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_agentx_results import cache_rates, comparison_row, compare
from util.agentx_reference import (
    accounting,
    benchmark_accounting,
    bmk_comparison,
    deduplicate_series,
    distribution,
    load_experiments,
    output_length_audit,
    request_metrics,
    router_metrics,
    sampling_audit,
    server_log,
    server_metrics,
    trace_lengths,
)


def record(
    *,
    phase="profiling",
    start=1_000_000_000,
    end=3_000_000_000,
    osl=3,
    error=None,
    cancelled=False,
    inner=None,
    turn=0,
    outer=0,
):
    return {
        "metadata": {
            "benchmark_phase": phase,
            "was_cancelled": cancelled,
            "source_trace_id": "trace",
            "source_outer_idx": outer,
            "source_inner_idx": inner,
            "turn_index": turn,
            "conversation_id": "trace::sa:child" if inner is not None else "trace",
            "x_correlation_id": "child" if inner is not None else "parent",
            "parent_correlation_id": "parent" if inner is not None else None,
            "request_start_ns": start,
            "request_end_ns": end,
        },
        "error": error,
        "metrics": {
            "request_latency": {"value": 2000, "unit": "ms"},
            "time_to_first_token": {"value": 500, "unit": "ms"},
            "input_sequence_length": {"value": 100, "unit": "tokens"},
            "output_sequence_length": {"value": osl, "unit": "tokens"},
            "osl_mismatch_diff_pct": {"value": 0, "unit": "%"},
        },
    }


class RequestReferenceTest(unittest.TestCase):
    def test_excludes_warmup_errors_cancellation_and_uses_observed_boundary(self):
        rows = [
            record(phase="warmup", start=0, end=99_000_000_000),
            record(error={"type": "error"}, end=80_000_000_000),
            record(cancelled=True, end=90_000_000_000),
            record(),
            record(start=2_000_000_000, end=5_000_000_000, osl=1),
        ]
        metrics = request_metrics(rows, 4)
        self.assertEqual(metrics["request_count"], 2)
        self.assertEqual(metrics["profile_duration_s"], 4)
        self.assertEqual(metrics["output_tput_per_gpu"], 0.25)
        self.assertEqual(metrics["input_tput_per_gpu"], 12.5)
        self.assertEqual(metrics["itl_s"]["count"], 1)
        self.assertEqual(metrics["itl_s"]["mean"], 0.75)

    def test_nanosecond_boundary_subtracts_before_float_conversion(self):
        r = record(start=1_700_000_000_000_000_001, end=1_700_000_000_000_000_004)
        self.assertEqual(request_metrics([r], 1)["profile_duration_s"], 3e-9)

    def test_missing_phase_is_historical_profiling(self):
        r = record()
        del r["metadata"]["benchmark_phase"]
        self.assertEqual(request_metrics([r], 1)["request_count"], 1)

    def test_unknown_unit_or_empty_success_pool_fails(self):
        r = record()
        r["metrics"]["request_latency"]["unit"] = "minutes"
        with self.assertRaises(ValueError):
            request_metrics([r], 1)
        with self.assertRaises(ValueError):
            request_metrics([record(error=True)], 1)

    def test_linear_percentile_and_population_std(self):
        d = distribution([1, 2, 3, 4])
        self.assertAlmostEqual(d["p90"], 3.7)
        self.assertAlmostEqual(d["std"], 1.25**0.5)
        self.assertIsNone(distribution([])["p90"])

    def test_cancellations_without_records_use_wire_log(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "benchmark.log"
            path.write_text(
                "Phase profiling timed out, cancelling all credits. Stats: sent=5, completed=3\n"
                "Phase profiling (profiling) complete | completed=3, cancelled=2, errors=1 | elapsed=3640.00s\n"
            )
            a = accounting(
                [record(), record(), record(error=True)], benchmark_accounting(path)
            )["profiling"]
            self.assertEqual(a["successful_records"], 2)
            self.assertEqual(a["error_records"], 1)
            self.assertEqual(a["cancelled_records"], 0)
            self.assertEqual(a["cancelled_without_record"], 2)
            self.assertTrue(a["wire_conservation"])

    def test_cancelled_error_record_is_not_double_counted(self):
        a = accounting([record(error=True, cancelled=True)], {})["profiling"]
        self.assertEqual(a["successful_records"], 0)
        self.assertEqual(a["error_records"], 0)
        self.assertEqual(a["cancelled_records"], 1)

    def test_original_identity_distinguishes_subagent_from_parent_turn(self):
        rows = [record(osl=20), record(inner=0, outer=1, osl=3, turn=0)]
        audit = output_length_audit(
            rows, {("trace", 0, None): 20, ("trace", 1, 0): 3}, {"trace": [20]}
        )
        self.assertEqual(audit["expected"]["mean"], 11.5)
        self.assertEqual(audit["historical_turn_index_lookup"]["mean"], 20)
        self.assertEqual(audit["mismatches"], [])
        self.assertEqual(audit["missing_sources"], [])

    def test_zero_osl_is_one_and_missing_mapping_stays_missing(self):
        audit = output_length_audit(
            [record(osl=1)], {("trace", 0, None): 0}, {"trace": [0]}
        )
        self.assertEqual(audit["expected"]["mean"], 1)
        self.assertEqual(audit["mismatches"], [])
        audit = output_length_audit([record()], {}, {})
        self.assertEqual(len(audit["missing_sources"]), 1)
        self.assertIsNone(audit["expected"]["mean"])

    def test_trace_mapping_preserves_outer_and_inner_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "t",
                        "requests": [
                            {"type": "s", "out": 20},
                            {"type": "subagent", "requests": [{"out": 3}, {"out": 4}]},
                            {"type": "s", "out": 10},
                        ],
                    }
                )
                + "\n"
            )
            lengths, legacy, order = trace_lengths(path)
            self.assertEqual(lengths[("t", 1, 1)], 4)
            self.assertEqual(lengths[("t", 2, None)], 10)
            self.assertEqual(legacy["t"], [20, 10])
            self.assertEqual(order, ["t"])


class ServerReferenceTest(unittest.TestCase):
    def series(self, endpoint, engine="0", total=100, **labels):
        return {
            "endpoint_url": endpoint,
            "labels": {"engine": engine, **labels},
            "stats": {"total": total},
        }

    def test_only_explicit_endpoint_aliases_deduplicate_same_engines(self):
        rows = [self.series("router"), self.series("engine"), self.series("other-node")]
        chosen, audit = deduplicate_series(
            rows, {"router": "node0", "engine": "node0"}, "engine"
        )
        self.assertEqual(sum(r["stats"]["total"] for r in chosen), 200)
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit[0]["selected_endpoint"], "engine")

    def test_different_engine_or_source_labels_are_not_deduplicated(self):
        rows = [
            self.series("engine", source="compute"),
            self.series("engine", source="cache"),
            self.series("engine", engine="1", source="cache"),
        ]
        chosen, _ = deduplicate_series(rows, {}, "engine")
        self.assertEqual(len(chosen), 3)

    def test_disagreeing_scrapes_preserve_both_and_prefer_direct(self):
        rows = [self.series("router", total=101), self.series("engine", total=100)]
        chosen, audit = deduplicate_series(
            rows, {"router": "node0", "engine": "node0"}, "engine"
        )
        self.assertEqual(chosen[0]["stats"]["total"], 100)
        self.assertEqual(audit[0]["stats_by_endpoint"]["router"]["total"], 101)

    def test_unidentifiable_or_duplicate_series_fail(self):
        with self.assertRaises(ValueError):
            deduplicate_series(
                [self.series("engine"), self.series("engine")], {}, "engine"
            )
        with self.assertRaises(ValueError):
            deduplicate_series(
                [{"endpoint_url": "engine", "labels": {}, "stats": {}}], {}, "engine"
            )

    def test_streamed_metrics_exclude_warmup_and_keep_all_rank_slices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "metrics.json"
            gauge = self.series("engine")
            gauge["stats"] = {"avg": 3, "max": 4}
            gauge["timeslices"] = [{"start_ns": 1, "end_ns": 2, "avg": 3}]
            source.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "vllm:prompt_tokens": {
                                "series": [self.series("engine"), self.series("router")]
                            },
                            "vllm:generation_tokens": {
                                "series": [self.series("engine", total=10)]
                            },
                            "vllm:num_requests_running": {"series": [gauge]},
                        },
                        "warmup_metrics": {
                            "vllm:prompt_tokens": {
                                "series": [self.series("engine", total=999)]
                            }
                        },
                    }
                )
            )
            result = server_metrics(
                source,
                {
                    "metrics_endpoint_groups": {"router": "node0", "engine": "node0"},
                    "preferred_metrics_endpoint": "engine",
                },
                root / "slices.jsonl",
            )
            self.assertEqual(result["counters"]["prompt_tokens"], 100)
            self.assertEqual(result["per_rank"]["0"]["num_requests_running"]["max"], 4)
            self.assertEqual(len((root / "slices.jsonl").read_text().splitlines()), 1)
            self.assertIn("prefix_cache_hits", result["missing_metrics"])

    def test_router_uses_worker_suffix_counts_one_of_two_log_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "router.log"
            path.write_text(
                "CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:parent' mapped to worker: http://localhost:8889@3\n"
                "Consistent hash routing: key='header:x-session-id:parent' -> worker='http://localhost:8889@3' (index=5)\n"
                "CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:child' mapped to worker: http://localhost:8889@1\n"
            )
            result = router_metrics(path, [record(), record(inner=0)])
            self.assertEqual(result["request_count"], 2)
            self.assertEqual(result["session_to_rank"], {"parent": 3, "child": 1})
            self.assertEqual(result["record_mapping_coverage"], 1)
            self.assertEqual(result["child_id_equal_parent_count"], 0)

    def test_server_worker_memory_and_engine_equivalent_tokens_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "server.log"
            path.write_text(
                "(Worker_DP3_EP3 pid=1) INFO Model loading took 125.88 GiB\n"
                "(Worker_DP3_EP3 pid=1) INFO Available KV cache memory: 32.27 GiB\n"
                "(EngineCore_DP3 pid=2) INFO GPU KV cache size: 2,742,953 tokens\n"
            )
            result = server_log(path, Path(temp) / "timeseries.jsonl")
            self.assertEqual(result["weight_gib_per_rank"]["3"], 125.88)
            self.assertEqual(result["available_kv_gib_per_rank"]["3"], 32.27)
            self.assertEqual(result["logged_equivalent_kv_tokens_total"], 2742953)

    def test_sampling_reports_lane_order_without_reordering_the_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "benchmark.log"
            path.write_text(
                "lane=01 sample_time= 53% root_next=3/4 trace_id=b\n"
                "lane=00 sample_time= 49% root_next=2/4 trace_id=a\n"
            )
            audit = sampling_audit(path, ["a", "b"], 42)
            self.assertEqual(audit["sequential_order_matches"], 2)
            self.assertEqual(audit["legacy_shuffled_order_matches"], 0)

    def test_server_log_keeps_samples_without_external_hit_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "server.log"
            path.write_text(
                "INFO 08-21 23:15:15 Reqs Running: 102, Waiting: 7, GPU KV cache used: 38.3%, Prefix cache hit rate: 0.0%\n"
            )
            output = Path(temp) / "timeseries.jsonl"
            result = server_log(path, output)
            self.assertEqual(result["scheduler_samples"], 1)
            row = json.loads(output.read_text())
            self.assertEqual(row["running"], 102)
            self.assertIsNone(row["external_hit_fraction_of_gpu_misses"])

    def test_server_log_deferred_field_does_not_drop_busy_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "server.log"
            path.write_text(
                "INFO 08-21 23:26:35 Reqs Running: 94, Waiting: 214, Deferred: 6 reqs, GPU KV cache used: 24.7%, Prefix cache hit rate: 49.2%, External prefix cache hit rate: 10.8%\n"
            )
            output = Path(temp) / "timeseries.jsonl"
            result = server_log(path, output)
            self.assertEqual(result["unparsed_scheduler_samples"], 0)
            row = json.loads(output.read_text())
            self.assertEqual(row["running"], 94)
            self.assertEqual(row["deferred"], 6)
            self.assertEqual(row["gpu_kv_fraction"], 0.247)


class ComparisonReferenceTest(unittest.TestCase):
    def inputs(self):
        exp = load_experiments(Path("data/experiments/agentx.yaml"))[0]
        metadata = {
            k: exp[k]
            for k in (
                "num_gpus",
                "hardware",
                "concurrency",
                "parallelism",
                "cache_backend",
                "model",
                "profiling_window_s",
                "trace_count",
            )
        }
        metadata.update(id=exp["id"], reference_id=exp["reference"]["id"], seed=0)
        result = {
            "agentx_metrics": {
                "concurrency": exp["concurrency"],
                "profile_duration_s": 4,
                "input_tokens": 640,
                "output_tokens": 64,
                "request_count": 2,
                "output_token_throughput_per_gpu_tps": 9999,
                "ttft_s": {"p90": 1},
                "tpot_s": {"p90": 0.1},
                "e2e_s": {"mean": 2},
            }
        }
        reference = {
            k: exp[k]
            for k in (
                "num_gpus",
                "hardware",
                "concurrency",
                "parallelism",
                "cache_backend",
                "model",
            )
        }
        reference.update(
            reference_id=exp["reference"]["id"],
            kv_cache_pool_tokens_semantics="logged equivalent tokens",
            metrics={
                "input_tput_per_gpu": 20,
                "output_tput_per_gpu": 2,
                "p90_ttft": 1,
                "p90_itl": 0.1,
                "mean_e2e": 2,
                "gpu_cache_hit_rate": 0.7,
                "cpu_cache_hit_rate": 0.2,
                "kv_cache_pool_tokens": 100000,
            },
        )
        return exp, result, reference, metadata

    def test_recomputes_both_per_gpu_throughputs_from_actual_gpu_count(self):
        exp, result, ref, metadata = self.inputs()
        row = comparison_row(exp, result, ref, metadata)
        self.assertEqual(row["output_tput_per_gpu"], 2)
        self.assertEqual(row["input_tput_per_gpu"], 20)
        self.assertEqual(row["output_tput_per_gpu_delta_pct"], 0)
        self.assertIsNone(row["cpu_hit_rate"])
        self.assertIsNone(row["gpu_hit_delta_pp"])
        self.assertIsNone(row["simulated_kv_capacity_tokens"])

    def test_different_gpu_count_topology_or_model_prevents_false_comparison(self):
        for key, value in (
            ("num_gpus", 64),
            ("model", "DeepSeek-V4-Proxy"),
            ("parallelism", {"tp": 8, "dp": 8, "ep": 8, "pp": 1}),
            ("profiling_window_s", 1800),
        ):
            exp, result, ref, metadata = self.inputs()
            metadata[key] = value
            row = comparison_row(exp, result, ref, metadata)
            self.assertFalse(row["comparable"])
            self.assertIn(key, row["incompatible_fields"])
            self.assertIsNone(row["output_tput_per_gpu_delta_pct"])
            if key == "num_gpus":
                self.assertEqual(row["output_tput_per_gpu"], 0.25)

    def test_offload_disabled_is_zero_missing_offload_metrics_are_not_zero(self):
        self.assertEqual(cache_rates({}, "none"), (None, 0))
        self.assertEqual(cache_rates({}, "simple_cpu"), (None, None))
        self.assertEqual(
            cache_rates(
                {"mooncake_cache_query_tokens": 10, "mooncake_local_gpu_hit_tokens": 6},
                "mooncake",
            ),
            (0.6, None),
        )

    def test_wrong_reference_and_unidentified_old_manifest_are_rejected(self):
        exp, result, ref, metadata = self.inputs()
        ref["reference_id"] = 123
        with self.assertRaises(ValueError):
            comparison_row(exp, result, ref, metadata)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "run_manifest.json"
            manifest.write_text('{"cases": []}')
            with self.assertRaises(ValueError):
                compare(root, Path("data/experiments/agentx.yaml"), manifest)

    def test_matrix_normalizes_each_experiment_including_four_gpu_points(self):
        exp, result, ref, metadata = self.inputs()
        for obj in (exp, ref, metadata):
            obj["num_gpus"] = 4
        ref["metrics"]["output_tput_per_gpu"] = 4
        row = comparison_row(exp, result, ref, metadata)
        self.assertEqual(row["output_tput_per_gpu"], 4)
        self.assertTrue(row["comparable"])
        self.assertEqual(row["output_tput_per_gpu_delta_pct"], 0)

    def test_manifest_drives_result_locations_and_duplicate_seeds_fail(self):
        exp, result, ref, metadata = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "custom").mkdir()
            (root / "custom" / "run.json").write_text(json.dumps(result))
            reference_file = root / exp["reference"]["metrics_file"]
            reference_file.parent.mkdir(parents=True)
            reference_file.write_text(json.dumps(ref))
            metadata["result_file"] = "custom/run.json"
            manifest = root / "run_manifest.json"
            manifest.write_text(json.dumps({"experiments": [metadata]}))
            report = compare(root, Path("data/experiments/agentx.yaml"), manifest)
            self.assertEqual(len(report["rows"]), 1)
            self.assertTrue(report["rows"][0]["comparable"])
            self.assertTrue((root / "comparison.csv").exists())
            manifest.write_text(json.dumps({"experiments": [metadata, metadata]}))
            with self.assertRaises(ValueError):
                compare(root, Path("data/experiments/agentx.yaml"), manifest)


if __name__ == "__main__":
    unittest.main()
