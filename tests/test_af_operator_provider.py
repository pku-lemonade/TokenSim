"""Af operator provider."""

from __future__ import annotations

from dataclasses import replace
import unittest

from TokenSim.af import AF_MATCH_TYPES, OperatorTableAFComputeProvider
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.lookup import MissingOperatorDataError
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from tests.support.af_hardware import af_test_hardware
from tests.support.af_queries import HIDDEN, _component_names, _compute_query, _moe


class OperatorTableComputeProviderTest(unittest.TestCase):
    def setUp(self):
        self.hardware = af_test_hardware(moe=_moe())
        self.model = self.hardware.model("TestModel")
        self.provider = OperatorTableAFComputeProvider(self.hardware, self.model)

    def test_stage_decomposition_matches_the_af_contract(self):
        a = self.provider.estimate(_compute_query("a"))
        router = self.provider.estimate(_compute_query("router"))
        expert = self.provider.estimate(_compute_query("expert"))
        self.assertEqual(
            _component_names(a),
            [
                "a_rmsnorm",
                "qkv_projection",
                "rope",
                "attention_core",
                "output_projection",
                "a_residual_add",
            ],
        )
        self.assertEqual(_component_names(router), ["f_rmsnorm", "router"])
        self.assertEqual(_component_names(expert), ["moe_expert", "f_residual_add"])
        for estimate in (a, router, expert):
            total = sum(item["latency_us"] for item in estimate.details["components"])
            self.assertAlmostEqual(estimate.service_duration, total / 1e6)
            self.assertGreater(estimate.service_duration, 0)
            # No measured package for the test devices: every operator is analytical.
            self.assertEqual(estimate.match_type, "analytical")
            self.assertIn(estimate.match_type, AF_MATCH_TYPES)
        self.assertEqual(a.dataset_version, "analytical:a-device")
        self.assertEqual(expert.dataset_version, "analytical:f-device")
        self.assertEqual(a.details["operator_backend"], "analytical")
        self.assertGreater(self.provider.stats_dict()["operator_query_count"], 0)

    def test_tensor_parallel_shrinks_rank_local_projections(self):
        a = self.provider.estimate(_compute_query("a", tp_size=2))
        keys = {item["name"]: item["key"] for item in a.details["components"]}
        # hidden 16, 8 heads of dim 2 -> q_dim 16, kv_dim 16; TP 2 halves both.
        self.assertEqual(
            (keys["qkv_projection"]["n"], keys["qkv_projection"]["k"]), (24, HIDDEN)
        )
        self.assertEqual(
            (keys["output_projection"]["n"], keys["output_projection"]["k"]),
            (HIDDEN, 8),
        )
        self.assertEqual(keys["attention_core"]["num_heads"], 4)
        self.assertEqual(keys["attention_core"]["num_kv_heads"], 4)
        self.assertEqual(keys["rope"]["hidden_size"], 16)

    def test_prefill_and_decode_group_attention_by_request_length(self):
        prefill = self.provider.estimate(
            _compute_query("a", phase="prefill", tokens=5, context=(2, 3))
        )
        attention = [
            item
            for item in prefill.details["components"]
            if item["name"] == "attention_core"
        ]
        self.assertEqual(
            [item["table"] for item in attention], ["context_attention"] * 2
        )
        self.assertEqual(
            [
                (item["key"]["input_seq_len"], item["key"]["batch_size"])
                for item in attention
            ],
            [(2, 1), (3, 1)],
        )
        decode = self.provider.estimate(
            _compute_query("a", tokens=3, context=(4, 8, 4))
        )
        attention = [
            item
            for item in decode.details["components"]
            if item["name"] == "attention_core"
        ]
        self.assertEqual(
            [item["table"] for item in attention], ["generation_attention"] * 2
        )
        self.assertEqual(
            [
                (item["key"]["context_len"], item["key"]["batch_size"])
                for item in attention
            ],
            [(4, 2), (8, 1)],
        )

    def test_moe_expert_uses_balanced_equivalent_tokens(self):
        expert = self.provider.estimate(
            _compute_query(
                "expert", tokens=6, ep_size=2, expert_workload=((0, 4), (1, 2))
            )
        )
        moe = next(
            item
            for item in expert.details["components"]
            if item["name"] == "moe_expert"
        )
        # 6 routed assignments on one of 2 EP ranks with top_k 2 -> 6 balanced input tokens.
        self.assertEqual(moe["key"]["num_tokens"], 6)
        self.assertEqual(moe["key"]["routed_assignments"], 6)
        self.assertEqual(moe["key"]["per_expert"], ((0, 4), (1, 2)))
        self.assertEqual(
            (moe["key"]["ep_size"], moe["key"]["top_k"], moe["key"]["num_experts"]),
            (2, 2, 4),
        )
        self.assertEqual(moe["key"]["inter_size"], 32)

    def test_dense_model_prices_ffn_gemms_on_the_f_rank(self):
        hardware = af_test_hardware()
        provider = OperatorTableAFComputeProvider(hardware, hardware.model("TestModel"))
        expert = provider.estimate(
            _compute_query("expert", tokens=2, tp_size=2, expert_workload=())
        )
        self.assertEqual(
            _component_names(expert),
            ["dense_gate_up", "dense_swiglu", "dense_down", "f_residual_add"],
        )
        keys = {item["name"]: item["key"] for item in expert.details["components"]}
        self.assertEqual(
            (keys["dense_gate_up"]["n"], keys["dense_gate_up"]["k"]), (64, HIDDEN)
        )
        self.assertEqual(
            (keys["dense_down"]["n"], keys["dense_down"]["k"]), (HIDDEN, 32)
        )
        router = provider.estimate(_compute_query("router"))
        self.assertEqual(_component_names(router), ["f_rmsnorm"])

    def test_invalid_local_work_and_expert_metadata_fail(self):
        with self.assertRaisesRegex(ValueError, "local_shape fields"):
            self.provider.estimate(_compute_query("a", local_shape={"tokens": 2}))
        with self.assertRaisesRegex(ValueError, "does not match model hidden_size"):
            self.provider.estimate(
                _compute_query(
                    "a",
                    local_shape={
                        "tokens": 2,
                        "hidden_size": 8,
                        "global_hidden_size": 8,
                        "tp_size": 1,
                    },
                )
            )
        with self.assertRaisesRegex(ValueError, "does not match local_token_count"):
            self.provider.estimate(
                _compute_query("expert", tokens=3, expert_workload=((0, 2),))
            )
        with self.assertRaisesRegex(ValueError, "unsupported AF compute stage"):
            self.provider.estimate(replace(_compute_query("a"), stage="other"))

    def test_measured_rows_take_precedence_and_table_only_rejects_gaps(self):
        source = SourceRecord("measured:test", "A", "measured")
        meta = PackageMeta(
            "test-v1", "a-device", "measured", sources={"measured:test": source}
        )
        package = OperatorDataPackage.from_rows(
            meta,
            {
                "gemm": [
                    {
                        "dtype": "fp16",
                        "m": 2,
                        "n": 48,
                        "k": HIDDEN,
                        "latency_us": 123.0,
                        "source_id": "measured:test",
                    }
                ]
            },
        )
        self.hardware.register_package(package)
        provider = OperatorTableAFComputeProvider(self.hardware, self.model)
        a = provider.estimate(_compute_query("a"))
        qkv = next(
            item for item in a.details["components"] if item["name"] == "qkv_projection"
        )
        self.assertEqual(
            (qkv["match_type"], qkv["latency_us"], qkv["source_id"]),
            ("exact", 123.0, "measured:test"),
        )
        self.assertEqual(a.dataset_version, "test-v1")
        # the remaining operators are still analytical, which is the composition's grade
        self.assertEqual(a.match_type, "analytical")
        self.assertTrue(a.source_id.startswith("compose:"))
        strict = OperatorTableAFComputeProvider(
            self.hardware, self.model, fallback="table_only"
        )
        with self.assertRaises(MissingOperatorDataError):
            strict.estimate(_compute_query("a"))
        with self.assertRaises(ConfigurationError):
            OperatorTableAFComputeProvider(
                self.hardware, self.model, fallback="table_only"
            ).resolver_for("f-device")
