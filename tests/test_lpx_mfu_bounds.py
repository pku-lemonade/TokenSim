"""Small analytic oracles for conditional pipeline timing projections."""

from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TokenSim.errors import ConfigurationError
from scripts.bound_lpx_mfu_layouts import (
    matrix_latency_bound,
    project,
    projected_frontier,
    saved_profile,
    zero_handoff_bound,
    width_coverage,
)
from scripts.lpx.replay_engine import ReplayNode
from scripts.lpx.replay_profiles import NativeProfile


def toy_profile():
    return NativeProfile(
        (
            ReplayNode((), ("model/worker:0",), "model/worker:0", 0, 3, 3),
            ReplayNode((0,), ("lp30:0:tx", "lp30:2:rx"), "lp30:0:tx", 0, 1, 1),
            ReplayNode((1,), (), None, 2, 0, 0),
            ReplayNode((0,), ("lp30:1:tx", "lp30:3:rx"), "lp30:1:tx", 0, 1, 1),
            ReplayNode((3,), (), None, 2, 0, 0),
            ReplayNode((2, 4), ("model/worker:2",), "model/worker:2", 0, 4, 4),
        ),
        dict(streaming=dict(fabric=dict(config=dict(routes=[])))),
    )


class MFUPartitionBoundTest(unittest.TestCase):
    def test_matrix_service_bound_keeps_precision_rates_and_serial_work(self):
        # Two requests share two chips: 120 / (2*3) linear seconds,
        # followed by 80 / (2*2.5) attention seconds on the same matrix units.
        rates = dict(batch=2, tp=2, fp8_peak=10, bf16_peak=5, gemm=0.3, attention=0.5)
        work = dict(linear=60, attention=40, total=100)
        self.assertEqual(matrix_latency_bound(work, **rates), 36)
        self.assertEqual(matrix_latency_bound(work, **(rates | dict(batch=1))), 18)
        for changes in (dict(batch=0), dict(tp=True), dict(attention=0), dict(gemm=2)):
            with self.assertRaises(ConfigurationError):
                matrix_latency_bound(work, **(rates | changes))

    def test_width_coverage_distinguishes_unmeasured_and_excluded_widths(self):
        family = dict(tp=2, kvp=1, kv_parallel_layers="full_attention")
        progress = dict(
            metric=dict(
                useful_flops_per_token=dict(linear=60, attention=40, total=100),
                fp8_peak_per_chip=10,
                bf16_peak_per_chip=5,
            ),
            shared_timing_config=dict(compute=dict(gemm_mfu=0.3, attention_mfu=0.5)),
            best_observed=dict(mfu_percent=20),
        )
        capacity = {
            (2, 1, "full_attention"): family | dict(maximum_resident_by_pp={"2": 4})
        }
        references = [family | dict(batch=1, zero_handoff_critical_path_s=120)]
        coverage = width_coverage(progress, capacity, references)
        self.assertEqual(coverage["widths"], 4)
        self.assertEqual(coverage["missing_profiles"], 1)
        self.assertEqual(coverage["excluded"], 3)
        self.assertEqual(coverage["rows"][1]["status"], "missing_profile")

    def test_explicit_memory_override_cannot_hide_compute_rate_changes(self):
        case = dict(tp=2, kvp=1, pp=2, batch=1, kv_parallel_layers="full_attention")
        shared = dict(
            compute=dict(gemm_mfu=0.3, attention_mfu=0.5), memory=dict(prefill_chunk=16)
        )
        row = dict(
            case=case,
            config=shared | dict(memory=dict(prefill_chunk=1)),
            profile_identity=dict(input_sha256="abc"),
            source_sha256="source",
        )
        progress = dict(
            timing_bounds=[row],
            observed_probes=[],
            shared_timing_config=shared,
            metric=dict(
                decode_context=101000,
                useful_flops_per_token=dict(linear=60, attention=40, total=100),
                fp8_peak_per_chip=10,
            ),
            best_observed=dict(mfu_percent=20),
            capacity_screen=dict(
                config=shared, rows=[case | dict(maximum_resident_by_pp={"2": 4})]
            ),
        )
        reference = dict(case=case, input_sha256="abc", zero_handoff_critical_path_s=30)
        with patch(
            "scripts.bound_lpx_mfu_layouts.analyze_reference", return_value=reference
        ):
            with self.assertRaises(ConfigurationError):
                project(progress, [], [Path("abc.json.gz")])
            result = project(
                progress, [], [Path("abc.json.gz")], allow_memory_changes=True
            )
            self.assertEqual(result["projected_groups"], 1)
            row["config"]["compute"] = dict(gemm_mfu=0.4, attention_mfu=0.5)
            with self.assertRaises(ConfigurationError):
                project(progress, [], [Path("abc.json.gz")], allow_memory_changes=True)

    def test_zero_handoff_relaxes_only_wire_and_flight(self):
        result = zero_handoff_bound(toy_profile(), tp=2, pp=2)
        self.assertEqual(result["critical_path_s"], 10)
        self.assertEqual(result["zero_handoff_critical_path_s"], 7)
        self.assertEqual(result["removed_critical_path_s"], 3)
        self.assertEqual(result["handoff_wires"], 2)
        self.assertEqual(result["removed_nodes"], 4)

    def test_routes_shape_and_inventory_cannot_silently_change_scope(self):
        original = toy_profile()
        metadata = deepcopy(original.metadata)
        metadata["streaming"]["fabric"]["config"]["routes"] = [{}]
        bad_flight = list(original.nodes)
        bad_flight[2] = replace(bad_flight[2], duration=1)
        bad_resource = list(original.nodes)
        bad_resource[1] = replace(
            bad_resource[1], resources=("lp30:0:vxm", "lp30:2:vxm")
        )
        for profile, tp, pp in (
            (replace(original, metadata=metadata), 2, 2),
            (replace(original, nodes=tuple(bad_flight)), 2, 2),
            (replace(original, nodes=tuple(bad_resource)), 2, 2),
            (original, 2, 1),
            (original, 1, 4),
        ):
            with self.assertRaises(ConfigurationError):
                zero_handoff_bound(profile, tp=tp, pp=pp)

    def test_projection_retains_complete_cohorts_and_all_allocated_chips(self):
        points = projected_frontier(
            dict(tp=2, maximum_resident_by_pp={"2": 3, "3": 5}),
            batch=2,
            latency_s=2,
            flops=8,
            peak_per_chip=10,
        )
        self.assertEqual(
            [row["optimistic_resident_requests"] for row in points], [2, 4]
        )
        self.assertEqual([row["chips"] for row in points], [4, 6])
        self.assertAlmostEqual(points[0]["mfu_percent_upper"], 20)
        self.assertAlmostEqual(points[1]["mfu_percent_upper"], 80 / 3)
        with self.assertRaises(ConfigurationError):
            projected_frontier({}, batch=1, latency_s=0, flops=1, peak_per_chip=1)

    def test_saved_profile_checks_input_and_node_body_fingerprints(self):
        data = [[[], [], None, 0, 2, 0, 0]]
        body = hashlib.sha256(
            json.dumps(data, separators=(",", ":")).encode()
        ).hexdigest()
        raw = dict(input_sha256="expected", body_sha256=body, nodes=data, metadata={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json.gz"
            with gzip.open(path, "wt") as stream:
                json.dump(raw, stream)
            self.assertEqual(saved_profile(path, "expected").nodes[0].duration, 2)
            with self.assertRaises(ConfigurationError):
                saved_profile(path, "other")
            raw["nodes"][0][4] = 3
            with gzip.open(path, "wt") as stream:
                json.dump(raw, stream)
            with self.assertRaises(ConfigurationError):
                saved_profile(path, "expected")


if __name__ == "__main__":
    unittest.main()
