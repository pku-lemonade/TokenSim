"""Ordinary and AF decode use one invocation for a context histogram."""

import unittest

from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.lookup import MissingOperatorDataError
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from tests.support.hardware import make_device, make_model


class RaggedDecodeTest(unittest.TestCase):
    def setUp(self):
        self.device, self.model = make_device(), make_model()
        self.backend = OperatorTableLatencyBackend(
            device=self.device,
            model=self.model,
            fallback="analytical_only",
            decode_context_bucket=1,
        )
        self.common = dict(
            attn_dtype="fp16",
            kv_cache_dtype="fp16",
            num_heads=8,
            num_kv_heads=8,
            head_dim=16,
            window_size=0,
        )

    def estimate(self, table, key):
        return self.backend.analytical.estimate(table, key, self.device)

    def test_step_preserves_ragged_work_and_charges_one_launch(self):
        contexts = list(range(16, 272, 16))
        requests = []
        for i, length in enumerate(contexts):
            req = Request(i, length - 1, 100, 16)
            req.generation_idx = 1
            req.num_computed_tokens = length - 1
            req.scheduled_tokens = 1
            requests.append(req)
        self.backend.estimate_step_latency(requests)
        import json

        expected = self.estimate(
            "ragged_generation_attention",
            {
                **self.common,
                "batch_size": len(contexts),
                "context_lengths": json.dumps([[n, 1] for n in contexts]),
            },
        )
        self.assertAlmostEqual(
            self.backend.stats.component_seconds["attention"],
            self.model.num_layers * expected.latency_us * 1e-6,
        )
        self.assertEqual(
            self.backend.stats.table_match_counts["ragged_generation_attention"][
                "analytical"
            ],
            1,
        )
        parts = [
            self.estimate(
                "generation_attention",
                {**self.common, "batch_size": 1, "context_len": n},
            )
            for n in contexts
        ]
        self.assertEqual(expected.work.kv_bytes, sum(p.work.kv_bytes for p in parts))
        self.assertEqual(expected.work.flops, sum(p.work.flops for p in parts))
        self.assertAlmostEqual(
            sum(p.latency_us for p in parts) - expected.latency_us,
            (len(contexts) - 1) * expected.overhead_us,
        )

    def package(self, table, key, latency):
        source = SourceRecord("measured", "A", "measured")
        return OperatorDataPackage.from_rows(
            PackageMeta(
                "ragged-v1", self.device.device_id, "test", sources={"measured": source}
            ),
            {
                table: [
                    {
                        **self.common,
                        **key,
                        "latency_us": latency,
                        "source_id": "measured",
                    }
                ]
            },
        )

    def measured_backend(self, package):
        return OperatorTableLatencyBackend(
            device=self.device, model=self.model, package=package, fallback="table_only"
        )

    def test_uniform_measurement_is_unchanged(self):
        package = self.package(
            "generation_attention", dict(batch_size=3, context_len=16), 7
        )
        backend = self.measured_backend(package)
        self.assertEqual(backend._attention_decode_batch({(16, 1): 3}), 7)

    def test_exact_ragged_measurement_and_histogram_order(self):
        package = self.package(
            "ragged_generation_attention",
            dict(batch_size=3, context_lengths="[[16,2],[32,1]]"),
            11,
        )
        backend = self.measured_backend(package)
        self.assertEqual(backend._attention_decode_batch({(32, 1): 1, (16, 1): 2}), 11)

    def test_strict_mode_cannot_sum_uniform_measurements_for_a_ragged_call(self):
        package = self.package(
            "generation_attention", dict(batch_size=3, context_len=16), 7
        )
        with self.assertRaises(MissingOperatorDataError):
            self.measured_backend(package)._attention_decode_batch(
                {(16, 1): 2, (32, 1): 1}
            )

    def test_verification_width_uses_context_work(self):
        actual = self.backend._attention_decode_batch({(64, 4): 2})
        expected = self.backend._attention_prefill(2, 4, 64)
        self.assertAlmostEqual(actual, expected)

    def test_empty_batch_has_no_query(self):
        self.assertEqual(self.backend._attention_decode_batch({}), 0)
        self.assertEqual(sum(self.backend.stats.match_type_counts.values()), 0)
