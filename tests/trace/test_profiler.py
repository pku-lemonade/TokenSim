"""Correlation ownership, ambiguity and comparison accounting regressions."""

import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from TokenSim.errors import ConfigurationError
from TokenSim.profiler.compare import compare_traces
from TokenSim.profiler.importer import read_trace
from TokenSim.profiler.matching import match_spans
from TokenSim.profiler.records import Span


def span(name, index, *, duration=1, shape=None):
    return Span(
        str(index),
        name,
        "0",
        0,
        0,
        "prefill",
        index * 10,
        index * 10 + duration,
        "compute",
        "fixture",
        "fixture",
        shape=shape or {},
    )


def fixtures(directory, *, durations=(5, 4)):
    directory = Path(directory)
    simulated = directory / "simulated.json"
    profile = directory / "profile_tp0.json.gz"
    sim = []
    observed = [
        dict(
            ph="X",
            name="tokensim.forward",
            cat="user_annotation",
            pid=1,
            tid=1,
            ts=1000,
            dur=100,
        )
    ]
    for index, (name, start, duration) in enumerate(
        zip(("qkv", "output"), (0, 5), durations)
    ):
        sim.append(
            dict(
                ph="X",
                name=name,
                cat="compute",
                pid=1,
                tid=1,
                ts=start,
                dur=5 if index == 0 else 4,
                args=dict(
                    node_id=f"forward:0:0:rank:0:{index}",
                    operation=name,
                    worker_id=0,
                    iteration_id=0,
                    layer_id=0,
                    phase="prefill",
                    token_count=1,
                ),
            )
        )
        observed.extend(
            [
                dict(
                    ph="X",
                    name=f"tokensim.layer.0.{name}",
                    cat="user_annotation",
                    pid=1,
                    tid=1,
                    ts=1010 + index * 20,
                    dur=10,
                ),
                dict(
                    ph="X",
                    name="aten::linear",
                    cat="cpu_op",
                    pid=1,
                    tid=1,
                    ts=1011 + index * 20,
                    dur=2,
                    args={"External id": index + 1},
                ),
                dict(
                    ph="X",
                    name="gemm_kernel",
                    cat="kernel",
                    pid=0,
                    tid=7,
                    ts=2000 + index * 7,
                    dur=duration,
                    args={"External id": index + 1, "device": 0, "stream": 7},
                ),
            ]
        )
    simulated.write_text(
        json.dumps(dict(traceEvents=sim, metadata={"model_id": "Fixture"}))
    )
    with gzip.open(profile, "wt") as stream:
        json.dump(dict(traceEvents=observed), stream)
    (directory / "manifest.json").write_text(
        json.dumps(dict(model="Fixture", batch_size=1, input_tokens=1))
    )
    return simulated, profile


class ProfilerComparisonTest(unittest.TestCase):
    def test_invalid_mapping_does_not_start_reading_inputs(self):
        with self.assertRaisesRegex(ConfigurationError, "object"):
            compare_traces("missing.json", [], "unused", mapping=False)

    def test_timing_only_partial_service_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            simulated, profile = fixtures(directory)
            content = json.loads(simulated.read_text())
            content["traceEvents"][0]["args"]["incomplete"] = True
            simulated.write_text(json.dumps(content))
            report = compare_traces(simulated, [profile], Path(directory) / "report")
            self.assertEqual(report["coverage"]["matched_units"], 1)
            self.assertTrue(
                any(item["operation"] == "qkv" for item in report["excluded"])
            )

    def test_real_cpu_and_gpu_annotations_share_one_forward(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "fixtures/profiler/vllm_qwen3_layer.json"
        )
        profile = read_trace(path)
        self.assertEqual({item.operation for item in profile.spans}, {"qkv", "rope"})
        self.assertTrue(
            all(item.forward == 0 and item.layer == 0 for item in profile.spans)
        )
        rope = next(item for item in profile.spans if item.operation == "rope")
        self.assertEqual(rope.details["operation_evidence"], "kernel_name")

    def test_real_row_parallel_scope_includes_communication(self):
        profile = (
            Path(__file__).resolve().parents[1]
            / "fixtures/profiler/vllm_qwen3_row_parallel.json"
        )
        imported = read_trace(profile)
        self.assertEqual(len(imported.spans), 1)
        self.assertEqual(
            imported.spans[0].details["simulated_members"], ["output", "all_reduce"]
        )
        self.assertTrue(imported.spans[0].details["communication_intervals"])
        with tempfile.TemporaryDirectory() as directory:
            simulated, _ = fixtures(directory)
            content = json.loads(simulated.read_text())
            for row, operation in zip(content["traceEvents"], ("output", "all_reduce")):
                row["args"]["operation"] = operation
            simulated.write_text(json.dumps(content))
            report = compare_traces(simulated, [profile], Path(directory) / "report")
            self.assertEqual(report["coverage"]["matched_units"], 1)
            accounting = report["accounting"]["profiled_by_clock"]["profile:0"]
            self.assertGreater(accounting["compute_busy_us"], 0)
            self.assertGreater(accounting["communication_busy_us"], 0)

    def test_observed_fusion_groups_across_a_layer_boundary(self):
        left = [span("ffn_residual", 0), span("attention_norm", 1)]
        left[1].layer = 1
        right = span("attention_norm", 2)
        right.layer = 1
        right.details["simulated_members"] = ["ffn_residual", "attention_norm"]
        pairs, unmatched, _ = match_spans(left, [right])
        self.assertEqual(pairs[0][0].members, ("0", "1"))
        self.assertFalse(unmatched)
        pairs, unmatched, _ = match_spans([left[1]], [right])
        self.assertFalse(pairs)
        self.assertEqual(len(unmatched), 2)

    def test_forward_rows_are_distinct_from_operator_rows(self):
        left = span("q_norm", 0, shape={"num_tokens": 1024, "forward_tokens": 32})
        right = span("q_norm", 1, shape={"forward_tokens": 32})
        pairs, unmatched, _ = match_spans([left], [right])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(unmatched)

    def test_correlation_uses_cpu_ownership_after_async_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            simulated, profile = fixtures(directory)
            imported = read_trace(profile)
            self.assertEqual(
                {item.operation for item in imported.spans}, {"qkv", "output"}
            )
            self.assertTrue(all(item.layer == 0 for item in imported.spans))
            report = compare_traces(simulated, [profile], Path(directory) / "report")
            self.assertEqual(report["coverage"]["matched_units"], 2)
            self.assertEqual(
                [row["service_error_us"] for row in report["matches"]], [0, 0]
            )
            self.assertEqual(report["matches"][1]["start_offset_us"], -2)
            self.assertTrue((Path(directory) / "report/operators.csv").is_file())
            self.assertTrue(
                (Path(directory) / "report/comparison.trace.json").is_file()
            )

    def test_repeated_or_crossing_matches_remain_ambiguous(self):
        for left, right in (
            ([span("qkv", 0), span("qkv", 1)], [span("qkv", 2, duration=100)]),
            ([span("qkv", 0), span("output", 1)], [span("output", 2), span("qkv", 3)]),
        ):
            pairs, unmatched, _ = match_spans(left, right)
            self.assertEqual(pairs, [])
            self.assertTrue(all(item["status"] == "ambiguous" for item in unmatched))

    def test_repeated_operations_with_unique_order_can_match(self):
        pairs, unmatched, _ = match_spans(
            [span("all_reduce", 0), span("all_reduce", 1)],
            [span("all_reduce", 2), span("all_reduce", 3)],
        )
        self.assertEqual(len(pairs), 2)
        self.assertFalse(unmatched)

    def test_explicit_fusion_rule_keeps_member_ids(self):
        pairs, unmatched, _ = match_spans(
            [span("attention_residual", 0), span("ffn_norm", 1)],
            [span("ffn_norm", 2)],
            {
                "groups": [
                    {
                        "simulated": ["attention_residual", "ffn_norm"],
                        "profiled": "ffn_norm",
                    }
                ]
            },
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].members, ("0", "1"))
        self.assertFalse(unmatched)

    def test_shape_conflicts_prevent_matching(self):
        pairs, unmatched, _ = match_spans(
            [span("qkv", 0, shape={"m": 1})], [span("qkv", 1, shape={"m": 4})]
        )
        self.assertFalse(pairs)
        self.assertEqual(len(unmatched), 2)

    def test_zero_duration_has_no_relative_error(self):
        with tempfile.TemporaryDirectory() as directory:
            simulated, profile = fixtures(directory, durations=(0, 4))
            report = compare_traces(simulated, [profile], Path(directory) / "report")
            self.assertIsNone(report["matches"][0]["relative_service_error"])
            self.assertEqual(
                report["matches"][0]["relative_error_unavailable_reason"],
                "zero_observed_duration",
            )

    def test_incompatible_model_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            simulated, profile = fixtures(directory)
            (Path(directory) / "manifest.json").write_text('{"model":"Different"}')
            with self.assertRaisesRegex(ConfigurationError, "model_id"):
                compare_traces(simulated, [profile], Path(directory) / "report")

    def test_unattributed_kernels_are_reported_as_unmatched(self):
        with tempfile.TemporaryDirectory() as directory:
            simulated, _ = fixtures(directory)
            profile = Path(directory) / "unknown.json"
            profile.write_text(
                json.dumps(
                    [
                        dict(
                            ph="X",
                            cat="kernel",
                            name="opaque",
                            ts=1,
                            dur=2,
                            pid=0,
                            tid=1,
                        )
                    ]
                )
            )
            report = compare_traces(simulated, [profile], Path(directory) / "report")
            self.assertEqual(report["status"], "no_matches")
            self.assertTrue(report["unmatched"])

    def test_begin_end_records_and_malformed_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            for rows in (
                [dict(ph="X", ts=0, dur=-1)],
                [dict(ph="B", ts=0)],
                [dict(ph="E", ts=1)],
                [dict(ph="B", ts="bad")],
                [dict(ph="X", ts=float("nan"), dur=1)],
            ):
                path.write_text(json.dumps(rows))
                with self.subTest(rows=rows), self.assertRaises(ConfigurationError):
                    read_trace(path)

    def test_offline_import_has_no_gpu_runtime_dependency(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import TokenSim.profiler.compare; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules",
            ],
            check=True,
        )
