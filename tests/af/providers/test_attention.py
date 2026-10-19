"""Exact variable-length work and model-owned Q/K normalization."""

from dataclasses import replace
import unittest

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.workload import work_for_table
from TokenSim.operator_data.lookup import MissingOperatorDataError
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import _compute_query


def ragged_key(lengths="[[2,2],[16,1]]", batch=3):
    return dict(
        attn_dtype="fp16",
        kv_cache_dtype="fp16",
        num_heads=8,
        num_kv_heads=8,
        head_dim=2,
        window_size=4,
        batch_size=batch,
        context_lengths=lengths,
    )


class AttentionInvocationTest(unittest.TestCase):
    def test_ragged_work_keeps_each_request_window_and_one_launch(self):
        key = ragged_key()
        actual = work_for_table("ragged_generation_attention", key)
        parts = [
            work_for_table(
                "generation_attention",
                {
                    **key,
                    "context_len": length,
                    "batch_size": count,
                },
            )
            for length, count in ((2, 2), (16, 1))
        ]
        self.assertEqual(actual.flops, sum(p.flops for p in parts))
        self.assertEqual(actual.kv_bytes, sum(p.kv_bytes for p in parts))
        self.assertEqual(actual.details["kernel_invocations"], 1)
        for lengths in (
            "[[2,2]]",
            "[[2,1.5],[16,1.5]]",
            "[[2.5,3]]",
            "[[true,3]]",
            "[]",
        ):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                work_for_table("ragged_generation_attention", ragged_key(lengths))

    def test_equal_kv_history_does_not_multiply_launch_cost(self):
        hardware = af_test_hardware()
        provider = OperatorTableAFComputeProvider(
            hardware, hardware.model("TestModel"), fallback="analytical_only"
        )
        uniform = provider.estimate(_compute_query(tokens=8, context=(16,) * 8))
        varied = provider.estimate(
            _compute_query(tokens=8, context=(12, 13, 14, 15, 17, 18, 19, 20))
        )
        core = [
            c for c in varied.details["components"] if c["name"] == "attention_core"
        ]
        self.assertEqual(len(core), 1)
        self.assertEqual(core[0]["table"], "ragged_generation_attention")
        self.assertAlmostEqual(uniform.service_duration, varied.service_duration)

    def test_strict_lookup_does_not_reuse_uniform_rows_for_ragged_batch(self):
        hardware = af_test_hardware()
        source = SourceRecord("measured:test", "A", "measured")
        package = OperatorDataPackage.from_rows(
            PackageMeta(
                "test", "a-device", "measured", sources={source.source_id: source}
            ),
            {
                "generation_attention": [
                    {
                        **ragged_key(),
                        "context_len": 8,
                        "latency_us": 10,
                        "source_id": source.source_id,
                    }
                ],
            },
        )
        hardware.register_package(package)
        resolver = OperatorTableAFComputeProvider(
            hardware, hardware.model("TestModel"), fallback="table_only"
        ).resolver_for("a-device")
        with self.assertRaises(MissingOperatorDataError):
            resolver.query("ragged_generation_attention", ragged_key())

    def test_qk_norm_is_a_model_property_for_both_protocols(self):
        hardware = af_test_hardware()
        model = replace(hardware.model("TestModel"), qk_norm=True)
        names = []
        for protocol in ("partial_return", "vllm_afd"):
            provider = OperatorTableAFComputeProvider(
                hardware, model, fallback="analytical_only", execution_protocol=protocol
            )
            names.append(
                [
                    c["name"]
                    for c in provider.estimate(_compute_query()).details["components"]
                ]
            )
        self.assertEqual(names[0], names[1])
        self.assertTrue({"q_norm", "k_norm"}.issubset(names[0]))
        self.assertEqual(
            model.attention_params_per_layer()
            - replace(model, qk_norm=False).attention_params_per_layer(),
            2 * model.head_dim,
        )
        self.assertTrue(model.to_dict()["qk_norm"])
        self.assertTrue(
            type(model)
            .from_mapping(
                dict(
                    model_id="qwen3-test",
                    hidden_size=16,
                    intermediate_size=64,
                    num_layers=1,
                    num_attention_heads=8,
                    qk_norm=True,
                )
            )
            .qk_norm
        )
        with self.assertRaises(ConfigurationError):
            replace(model, qk_norm=1)
