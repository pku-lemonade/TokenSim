"""Physical attention growth and multi-axis measured-boundary regressions."""

import math
from dataclasses import replace
from unittest import TestCase

from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.device import SourcedValue
from TokenSim.latency.attention_work import (
    AttentionWorkConfig,
    CompressedAttentionWork,
    causal_limited_pairs,
    compressed_context_pairs,
)
from TokenSim.operator_data.lookup import LookupPolicy, OperatorLookup
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.operator_data.schema import TABLE_SPECS


class AttentionSOLTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = HardwareContext.load("data")
        cls.model = cls.context.model("DeepSeek-V4-Pro")
        cls.device = cls.context.device("b200")
        cls.package = cls.context.operator_package(cls.device.device_id, "vllm")
        cls.sol = CompressedAttentionWork(cls.model, cls.device)
        cls.row = next(
            row for row in cls.package.tables["dsv4_attention"]
            if row["layer_kind"] == "csa" and row["phase"] == "generation"
            and row["tp_size"] == 1 and row["batch_size"] == 64
            and row["context_len"] == 16384
        )

    def key(self, **changes):
        return {**{name: self.row[name] for name in TABLE_SPECS["dsv4_attention"].key_fields}, **changes}

    def test_causal_pair_counts_match_explicit_short_sequences(self):
        for queries in (1, 4, 129, 513):
            for prefix in (0, 1, 127, 128, 4095, 8192):
                self.assertEqual(
                    causal_limited_pairs(3, queries, prefix, 128),
                    3 * sum(min(prefix + i + 1, 128) for i in range(queries)),
                )
                for ratio, limit in ((4, 1024), (128, 100)):
                    self.assertEqual(
                        compressed_context_pairs(3, queries, prefix, ratio, limit),
                        3 * sum(min((prefix + i + 1) // ratio, limit) for i in range(queries)),
                    )

    def test_context_growth_preserves_fixed_work_and_sparse_caps(self):
        short = self.sol.work(self.key())
        long = self.sol.work(self.key(context_len=1048576))
        for name in ("projection", "output_absorption", "compressor", "indexer_weights", "attention"):
            self.assertEqual(short.flops[name], long.flops[name], name)
        for name in short.bytes:
            if name != "indexer_kv":
                self.assertEqual(short.bytes[name], long.bytes[name], name)
        self.assertGreater(long.flops["indexer"], short.flops["indexer"])
        self.assertGreater(long.bytes["indexer_kv"], short.bytes["indexer_kv"])
        self.assertLess(long.latency_us / short.latency_us, 1048576 / 16384)
        hca_short = self.sol.work(self.key(layer_kind="hca", compress_ratio=128))
        hca_long = self.sol.work(self.key(layer_kind="hca", compress_ratio=128, context_len=1048576))
        self.assertEqual(hca_long.bytes["indexer_kv"], 0)
        self.assertGreater(hca_long.bytes["attention_kv"], hca_short.bytes["attention_kv"])

    def test_batch_growth_does_not_repeat_projection_weights(self):
        one = self.sol.work(self.key(batch_size=1))
        many = self.sol.work(self.key(batch_size=64))
        for name, value in one.flops.items():
            self.assertEqual(many.flops[name], value * 64, name)
        self.assertEqual(one.bytes["projection_weights"], many.bytes["projection_weights"])
        self.assertEqual(many.bytes["attention_kv"], one.bytes["attention_kv"] * 64)

    def test_layout_parameters_change_reads_without_changing_flops(self):
        original = self.sol.work(self.key(context_len=1048576))
        serving = CompressedAttentionWork(
            self.model, self.device, AttentionWorkConfig(kv_state_bytes=512, indexer_state_bytes=68)
        ).work(self.key(context_len=1048576))
        self.assertEqual(original.flops, serving.flops)
        self.assertLess(serving.bytes["attention_kv"], original.bytes["attention_kv"])
        self.assertLess(serving.bytes["indexer_kv"], original.bytes["indexer_kv"])
        self.assertLess(serving.memory_seconds, original.memory_seconds)

    def test_growth_fast_path_matches_explicit_flop_byte_accounting(self):
        for ratio in (1, 4, 128):
            for tp in (1, 4, 8):
                for phase, queries, context in (("generation", 1, 1), ("generation", 1, 1048576),
                                                ("context", 512, 0), ("context", 4096, 62464)):
                    key = self.key(compress_ratio=ratio, tp_size=tp, num_heads=128 // tp,
                                   phase=phase, input_seq_len=queries, context_len=context)
                    self.assertAlmostEqual(self.sol.estimate_us(key), self.sol.work(key).latency_us, places=8)

    def test_kv_reuse_is_neutral_when_query_math_is_the_bound(self):
        device = replace(self.device, memory_bandwidth_bytes_per_s=SourcedValue(1e20, "fixture", "D"))
        sol = CompressedAttentionWork(self.model, device)
        key = self.key(context_len=1048576)
        self.assertEqual(sol.estimate_us(key, indexer_kv_fraction=0.25), sol.estimate_us(key))

    def test_shared_work_depends_on_rows_and_geometry_not_phase_or_context(self):
        for ratio in (1, 4, 128):
            for tp in (1, 4, 8):
                with self.subTest(ratio=ratio, tp=tp):
                    base = self.key(compress_ratio=ratio, tp_size=tp, num_heads=128 // tp)
                    shared, phase = self.sol.shared_phase_us({**base, "batch_size": 512, "context_len": 16384})
                    context_shared, context_phase = self.sol.shared_phase_us(
                        {**base, "phase": "context", "batch_size": 1,
                         "input_seq_len": 512, "context_len": 131072})
                    self.assertEqual(shared, context_shared)
                    self.assertGreater(shared, 0)
                    self.assertGreaterEqual(phase, 0)
                    self.assertGreaterEqual(context_phase, 0)
                    self.assertGreaterEqual(shared + phase, self.sol.estimate_us({**base, "batch_size": 512}))

    def test_shared_phase_reuse_changes_only_phase_bound(self):
        key = self.key(context_len=1048576)
        common, phase = self.sol.shared_phase_us(key)
        shared_after, phase_after = self.sol.shared_phase_us(key, indexer_kv_fraction=0.5)
        self.assertEqual(shared_after, common)
        self.assertLessEqual(phase_after, phase)
        self.assertGreaterEqual(phase_after, 0)

    def test_indexer_reuse_retains_hca_and_per_query_sparse_kv_work(self):
        key = self.key(context_len=131072)
        work = self.sol.work(key)
        math_us, memory_us, indexer_bytes = self.sol.limits_us(key)
        actual = self.sol.estimate_us(key, indexer_kv_fraction=0.25)
        expected = max(math_us, (sum(work.bytes.values()) - 0.75 * work.bytes["indexer_kv"]) / self.sol.bandwidth * 1e6)
        self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(memory_us, work.memory_seconds * 1e6)
        self.assertEqual(indexer_bytes, work.bytes["indexer_kv"])
        hca = self.key(layer_kind="hca", compress_ratio=128, context_len=131072)
        self.assertEqual(self.sol.estimate_us(hca, indexer_kv_fraction=0.25), self.sol.estimate_us(hca))

    def test_joint_extrapolation_preserves_one_boundary_efficiency(self):
        package = OperatorDataPackage.from_rows(self.package.meta, {"dsv4_attention": [self.row]})
        lookup = OperatorLookup(
            package, LookupPolicy(extrapolate="analytical", max_extrapolation_ratio=64),
            analytical_scaler=lambda _, key: self.sol.work(key).latency_us,
            bind_analytical_axes=True,
        )
        target = self.key(batch_size=128, context_len=32768)
        prediction = lookup.lookup("dsv4_attention", target)
        expected = self.row["latency_us"] * self.sol.work(target).latency_us / self.sol.work(self.key()).latency_us
        self.assertAlmostEqual(prediction.latency_us, expected)
        self.assertEqual(prediction.match_type, "extrapolated")

    def test_ragged_interpolation_grows_each_measured_batch_slice(self):
        low = {**self.row, "batch_size": 32, "latency_us": 150.0}
        high = {**self.row, "batch_size": 64, "latency_us": 180.0}
        package = OperatorDataPackage.from_rows(self.package.meta, {"dsv4_attention": [low, high]})
        lookup = OperatorLookup(
            package, LookupPolicy(extrapolate="analytical", max_extrapolation_ratio=64),
            analytical_scaler=lambda _, key: self.sol.work(key).latency_us,
            bind_analytical_axes=True,
        )
        target = self.key(batch_size=48, context_len=32768)
        predictions = [
            row["latency_us"] * self.sol.work({**row, "context_len": 32768}).latency_us / self.sol.work(row).latency_us
            for row in (low, high)
        ]
        weight = math.log(48 / 32) / math.log(64 / 32)
        expected = predictions[0] + weight * (predictions[1] - predictions[0])
        self.assertAlmostEqual(lookup.lookup("dsv4_attention", target).latency_us, expected)

    def test_measured_values_remain_exact(self):
        lookup = OperatorLookup(
            self.package, LookupPolicy(extrapolate="analytical", max_extrapolation_ratio=64),
            analytical_scaler=lambda _, key: self.sol.work(key).latency_us,
            bind_analytical_axes=True,
        )
        rows = self.package.tables["dsv4_attention"]
        for row in rows[::71]:
            actual = lookup.lookup("dsv4_attention", row)
            self.assertEqual(actual.match_type, "exact")
            self.assertEqual(actual.latency_us, row["latency_us"])


if __name__ == "__main__":
    import unittest

    unittest.main()
