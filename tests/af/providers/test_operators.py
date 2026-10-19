"""Operator ownership, precision and attention/FFN decomposition."""

from __future__ import annotations

from dataclasses import replace
import unittest
from tests.af.support.config import _with_prefill_pool

import simpy

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.af.providers.queries import AF_MATCH_TYPES
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.simulation import SimulationConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceCatalog
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.operator_data.lookup import MissingOperatorDataError
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from TokenSim.operator_data.workload import elementwise_work
from tests.af.support.config import _cluster
from tests.af.support.execution import af_test_hardware
from tests.af.support.model import HIDDEN, _component_names, _compute_query, _moe


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


class AFProviderCorrectnessTest(unittest.TestCase):
    def setUp(self):
        hardware = af_test_hardware(moe=_moe())
        hardware = replace(
            hardware,
            devices=DeviceCatalog(
                [
                    replace(
                        device,
                        peak_compute={
                            **device.peak_compute,
                            "fp8": device.peak_compute["fp16"],
                        },
                    )
                    for device in hardware.devices
                ]
            ),
        )
        self.provider = OperatorTableAFComputeProvider(
            hardware, hardware.model("TestModel"), fallback="analytical_only"
        )

    def test_runtime_router_prices_topk_without_repeating_attention_norm(self):
        query = replace(_compute_query("router"), layer_type="moe")
        components = self.provider.estimate(query).details["components"]
        self.assertEqual([c["name"] for c in components], ["router", "router_topk"])
        self.assertEqual(components[1]["key"]["op_name"], "topk")

    def test_routed_experts_do_not_own_residual_addition(self):
        query = replace(
            _compute_query("expert"), stage="routed_expert", layer_type="moe"
        )
        components = self.provider.estimate(query).details["components"]
        self.assertEqual([c["name"] for c in components], ["moe_expert"])

    def test_combine_and_residual_use_the_owned_hidden_interval(self):
        for stage in ("weighted_combine", "residual"):
            query = replace(
                _compute_query("a", tp_size=2),
                stage=stage,
                operation=stage,
                layer_type="moe",
            )
            components = self.provider.estimate(query).details["components"]
            self.assertEqual(len(components), 1)
            self.assertEqual(components[0]["key"]["hidden_size"], 8)

    def test_weight_and_kv_dtypes_are_independent_of_activations(self):
        query = replace(
            _compute_query("a", dtype="bf16", context=(8192,)),
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="fp32",
        )
        estimate = self.provider.estimate(query)
        components = estimate.details["components"]
        for component in components:
            if component["table"] == "gemm":
                self.assertEqual(component["key"]["dtype"], "fp8")
            elif component["table"] == "elementwise":
                self.assertEqual(component["key"]["dtype"], "bf16")
            elif component["table"] == "generation_attention":
                self.assertEqual(component["key"]["kv_cache_dtype"], "fp32")
        smaller_kv = self.provider.estimate(replace(query, kv_cache_dtype="fp16"))
        self.assertGreater(estimate.service_duration, smaller_kv.service_duration)

    def test_router_and_expert_queries_use_weight_precision(self):
        for stage in ("router", "expert"):
            query = replace(_compute_query(stage, dtype="bf16"), weight_dtype="fp8")
            components = self.provider.estimate(query).details["components"]
            weights = [
                c for c in components if c["table"] in {"gemm", "moe", "routed_experts"}
            ]
            self.assertTrue(weights)
            self.assertTrue(all(c["key"]["dtype"] == "fp8" for c in weights))

    def test_prefill_reads_cached_keys_in_addition_to_new_queries(self):
        query = replace(
            _compute_query("a", phase="prefill", tokens=8, context=(8,)),
            query_lengths=(8,),
        )
        plain = self.provider.estimate(query)
        cached = self.provider.estimate(replace(query, context_lengths=(8192,)))
        self.assertGreater(cached.service_duration, plain.service_duration)
        attention = next(
            c for c in cached.details["components"] if c["name"] == "attention_core"
        )
        self.assertEqual(attention["key"]["input_seq_len"], 8)
        self.assertEqual(attention["key"]["context_len"], 8192)

    def test_uneven_prefill_queries_are_not_replaced_by_their_mean(self):
        query = replace(
            _compute_query("a", phase="prefill", tokens=7, context=(12, 32)),
            query_lengths=(2, 5),
        )
        components = self.provider.estimate(query).details["components"]
        attention = [c for c in components if c["name"] == "attention_core"]
        self.assertEqual(
            [(c["key"]["input_seq_len"], c["key"]["context_len"]) for c in attention],
            [(2, 12), (5, 32)],
        )

    def test_prefill_rejects_queries_past_available_keys(self):
        query = replace(
            _compute_query("a", phase="prefill", tokens=8, context=(4,)),
            query_lengths=(8,),
        )
        with self.assertRaisesRegex(ValueError, "KV contexts"):
            self.provider.estimate(query)


class PrecisionConversionTest(unittest.TestCase):
    def test_cast_accounts_for_source_reads_and_target_writes(self):
        for source, target, byte_width in (("bf16", "fp8", 3), ("fp8", "fp32", 5)):
            work = elementwise_work(f"cast_{source}_to_{target}", 2, 16, target)
            self.assertEqual(work.activation_bytes, 32 * byte_width)
            self.assertGreater(work.flops, 0)
        with self.assertRaisesRegex(ValueError, "target dtype"):
            elementwise_work("cast_bf16_to_fp8", 1, 16, "bf16")

    def test_builtin_provider_executes_wire_codecs_with_tp_shards(self):
        for tp in (1, 2):
            with self.subTest(tp=tp):
                cluster = _cluster(a_tp=tp, a_dp=1, f_tp=tp, f_dp=1, f_ep=1)
                cluster = replace(
                    cluster,
                    af=replace(
                        cluster.af,
                        latency_provider="operator_table",
                        activation_dtype="bf16",
                        links=[
                            replace(
                                cluster.af.links[0],
                                a_to_f_dtype="fp8",
                                f_to_a_dtype="fp32",
                            )
                        ],
                    ),
                )
                hardware = af_test_hardware(capacity_gib=0.1)
                hardware = replace(
                    hardware,
                    devices=DeviceCatalog(
                        [
                            replace(
                                device,
                                peak_compute={
                                    **device.peak_compute,
                                    "fp32": device.peak_compute["fp16"],
                                },
                            )
                            for device in hardware.devices
                        ]
                    ),
                )
                env = simpy.Environment()
                engine = LLMEngine(
                    env=env,
                    kv_transfer_config=KVTransferConfig.default(),
                    model_spec=hardware.model("TestModel"),
                    cluster_config=_with_prefill_pool(cluster),
                    hardware=hardware,
                    config=SimulationConfig(
                        block_size=4, latency_fallback="analytical_only"
                    ),
                )
                req = Request(1, 8, 4, 4)
                req.generation_idx = 1
                executor = engine.af_executor
                iteration = executor.submit(0, [req])
                env.run(until=env.process(executor.run_iteration(iteration)))
                codecs = [
                    e for e in executor.trace.events if e.stage in {"encode", "decode"}
                ]
                self.assertTrue(codecs)
                self.assertTrue(all(e.service_duration > 0 for e in codecs))
                operations = {
                    e.details["components"][0]["key"]["op_name"] for e in codecs
                }
                self.assertEqual(
                    operations,
                    {
                        "cast_bf16_to_fp8",
                        "cast_fp8_to_bf16",
                        "cast_bf16_to_fp32",
                        "cast_fp32_to_bf16",
                    },
                )
                executor.trace.validate_invariants(iteration.iteration_id)
