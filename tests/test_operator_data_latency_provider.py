import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import simpy
import yaml

from tests.test_af_executor import AFRooflineStub, LightweightCacheConfig, _psla_config
from tests.test_af_parallel_planning import _cluster
from TokenSim.af import (
    AFComputeQuery,
    AFEndpoint,
    AFExecutor,
    AFRoutingEdge,
    AFTensorSlice,
    AFTransferQuery,
    compile_af_runtime_plan,
    resolve_af_latency_providers,
)
from TokenSim.config.cluster_config import AFConfig, AFEffectiveDTypes
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, reset_g_time
from TokenSim.moe.config import MoEModelConfig
from TokenSim.errors import SimulationStateError
from TokenSim.operator_data import (
    Calibration,
    EvidenceGrade,
    HardwareProfile,
    LookupResult,
    MissingOperatorDataError,
    ModelOperatorManifest,
    Observation,
    OperatorDataComputeLatencyProvider,
    OperatorDataGenerator,
    OperatorDataLookup,
    OperatorDataPackage,
    OperatorDataTransferLatencyProvider,
    OperatorDataValidationError,
    SourceRecord,
    ValidationReport,
    estimate_operator,
)
from TokenSim.operator_data.package import _read_parquet
from TokenSim.operator_data.providers import ModelShape
from util.request import LLMSource


def _tempdir():
    return tempfile.TemporaryDirectory()


def _hardware(device):
    return HardwareProfile(
        device=device,
        backend="analytical",
        peak_compute_by_dtype={"bf16": 1e12, "fp8": 2e12},
        memory_bandwidth_bytes_per_s=1e11,
        memory_capacity_bytes=230_000_000,
        interconnect_bandwidth_bytes_per_s=480_000_000_000,
    )


def _key(table, device="a-device", **overrides):
    values = {
        "gemm": {
            "device": device,
            "backend": "analytical",
            "weight_dtype": "bf16",
            "activation_dtype": "bf16",
            "m": 2,
            "n": 24,
            "k": 8,
        },
        "context_attention": {
            "device": device,
            "backend": "analytical",
            "attention_type": "mha",
            "activation_dtype": "bf16",
            "kv_cache_dtype": "bf16",
            "batch_size": 1,
            "input_seq_len": 2,
            "num_heads": 2,
            "num_kv_heads": 2,
            "head_dim": 4,
            "window_size": 0,
        },
        "generation_attention": {
            "device": device,
            "backend": "analytical",
            "attention_type": "mha",
            "activation_dtype": "bf16",
            "kv_cache_dtype": "bf16",
            "batch_size": 1,
            "context_len": 2,
            "num_heads": 2,
            "num_kv_heads": 2,
            "head_dim": 4,
            "window_size": 0,
        },
        "moe": {
            "device": device,
            "backend": "analytical",
            "operation": "routed_expert",
            "weight_dtype": "bf16",
            "activation_dtype": "bf16",
            "phase": "decode",
            "distribution": "balanced",
            "num_tokens": 2,
            "hidden_size": 8,
            "inter_size": 16,
            "top_k": 2,
            "num_experts": 4,
            "tp_size": 1,
            "ep_size": 1,
        },
        "elementwise": {
            "device": device,
            "backend": "analytical",
            "op_name": "rmsnorm",
            "activation_dtype": "bf16",
            "source_dtype": "bf16",
            "target_dtype": "bf16",
            "num_tokens": 2,
            "hidden_size": 8,
        },
        "communication": {
            "source_device": "a-device",
            "target_device": "f-device",
            "backend": "analytical",
            "network": "af-net",
            "link_group": "af-fabric",
            "resource_lane": "a_to_f",
            "operation": "af_a_to_f",
            "direction": "a_to_f",
            "transfer_dtype": "bf16",
            "source_endpoint": "a",
            "target_endpoint": "f",
            "group_size": 2,
            "fixed_latency_us": 1.0,
            "effective_bandwidth_bytes_per_s": 16_000_000.0,
        },
    }[table]
    values.update(overrides)
    return values


def _fixture_inputs():
    source = SourceRecord(
        source_id="measured:test",
        evidence_grade=EvidenceGrade.A,
        method="measured",
        device="test-fixture",
        backend="analytical",
        reference="deterministic test measurement",
        input_hash="fixture-v1",
    )
    shapes = []
    observations = []

    def add(table, key, latency):
        shapes.append({"table": table, "key": key})
        observations.append(
            Observation(table, key, latency, source.source_id, split="runtime")
        )

    # A-side attention/Router GEMMs and one interpolation shape.
    for tokens in (1, 2):
        add("gemm", _key("gemm", m=tokens, n=24, k=8), 10.0 * tokens)
        add("gemm", _key("gemm", m=tokens, n=8, k=8), 8.0 * tokens)
        add(
            "gemm",
            _key("gemm", device="a-device", m=tokens, n=4, k=8),
            4.0 * tokens,
        )
    add("gemm", _key("gemm", m=1, n=7, k=8), 10.0)
    add("gemm", _key("gemm", m=3, n=7, k=8), 30.0)

    for seq, latency in ((1, 5.0), (2, 10.0), (3, 15.0)):
        add(
            "context_attention",
            _key("context_attention", input_seq_len=seq),
            latency,
        )
        add(
            "generation_attention",
            _key("generation_attention", context_len=seq),
            latency,
        )
    # TP=2 A-rank local shapes for an exact two-token query.
    add("gemm", _key("gemm", m=2, n=12, k=8), 12.0)
    add("gemm", _key("gemm", m=2, n=8, k=4), 7.0)
    add(
        "generation_attention",
        _key(
            "generation_attention",
            context_len=2,
            num_heads=1,
            num_kv_heads=1,
        ),
        6.0,
    )
    for phase in ("prefill", "decode"):
        for tokens in (1, 2, 3, 4):
            for top_k, num_experts in ((2, 4), (1, 1)):
                add(
                    "moe",
                    _key(
                        "moe",
                        device="f-device",
                        phase=phase,
                        num_tokens=tokens,
                        top_k=top_k,
                        num_experts=num_experts,
                    ),
                    6.0 * tokens,
                )
            add(
                "moe",
                _key(
                    "moe",
                    device="f-device",
                    operation="shared_expert",
                    phase=phase,
                    num_tokens=tokens,
                    top_k=1,
                    num_experts=1,
                ),
                7.0 * tokens,
            )
    for tokens in (1, 3):
        add(
            "moe",
            _key(
                "moe",
                device="f-device",
                phase="decode",
                num_tokens=tokens,
                top_k=1,
                num_experts=1,
                tp_size=2,
                ep_size=2,
            ),
            10.0 * tokens,
        )
    for device in ("a-device", "f-device"):
        for tokens in (1, 2, 3, 4):
            for op, latency in (
                ("rmsnorm", 1.0),
                ("rope", 1.0),
                ("residual_add", 1.0),
                ("data_reorder", 1.0),
                ("swiglu", 1.0),
                ("routing_weight_multiply", 1.0),
                ("routed_reduce", 1.0),
                ("shared_sum", 1.0),
            ):
                add(
                    "elementwise",
                    _key(
                        "elementwise",
                        device=device,
                        op_name=op,
                        num_tokens=tokens,
                    ),
                    latency * tokens,
                )
    for op, latency in (
        ("rmsnorm", 1.0),
        ("rope", 1.0),
        ("residual_add", 1.0),
        ("data_reorder", 1.0),
        ("swiglu", 1.0),
        ("routing_weight_multiply", 1.0),
        ("routed_reduce", 1.0),
        ("shared_sum", 1.0),
    ):
        add(
            "elementwise",
            _key(
                "elementwise",
                device="a-device",
                op_name=op,
                num_tokens=2,
                hidden_size=4,
            ),
            latency,
        )

    for device in ("a-device", "f-device"):
        for op_name, source_dtype, target_dtype in (
            ("encode", "bf16", "fp8"),
            ("decode", "fp8", "bf16"),
        ):
            for element_count in (4, 8, 16):
                add(
                    "elementwise",
                    _key(
                        "elementwise",
                        device=device,
                        op_name=op_name,
                        activation_dtype="bf16",
                        source_dtype=source_dtype,
                        target_dtype=target_dtype,
                        num_tokens=1,
                        hidden_size=element_count,
                    ),
                    float(element_count),
                )

    for operation, direction, source_device, target_device, group_size in (
        ("af_a_to_f", "a_to_f", "a-device", "f-device", 2),
        ("af_f_to_a", "f_to_a", "f-device", "a-device", 2),
    ):
        key = _key(
            "communication",
            operation=operation,
            source_device=source_device,
            target_device=target_device,
            direction=direction,
            resource_lane=direction,
            source_endpoint=("a" if source_device == "a-device" else "f"),
            target_endpoint=("a" if target_device == "a-device" else "f"),
            group_size=group_size,
        )
        add("communication", key, 0.0)
    for operation in ("all_reduce", "all_gather", "reduce_scatter"):
        for source_device, target_device in (
            ("a-device", "a-device"),
            ("f-device", "f-device"),
        ):
            add(
                "communication",
                _key(
                    "communication",
                    operation=operation,
                    direction="collective",
                    source_device=source_device,
                    target_device=target_device,
                    resource_lane="collective",
                    source_endpoint=(
                        "a" if source_device == "a-device" else "f"
                    ),
                    target_endpoint=(
                        "a" if target_device == "a-device" else "f"
                    ),
                    group_size=2,
                ),
                0.0,
            )
    # Whole-execution observations deliberately differ from singleton sums.
    for heads in (1, 2):
        add("generation_attention", _key("generation_attention", batch_size=2,
            context_len=2, num_heads=heads, num_kv_heads=heads), 7.0)
    for phase in ("prefill", "decode"):
        for counts in ((1, 1), (1, 2), (2, 1), (2, 2), (1, 1, 1, 1)):
            add("moe_group", dict(_key("moe", device="f-device", phase=phase,
                num_tokens=sum(counts), top_k=1, num_experts=len(counts)),
                expert_token_counts=",".join(map(str, counts)),
                execution_boundary="unweighted_partials"), 5.0 + sum(counts))
    return source, shapes, observations


def _write_manifest(path, duplicate_includes=False):
    includes = {}
    if duplicate_includes:
        includes = {
            "qkv_projection": ["projection"],
            "output_projection": ["projection"],
        }
    document = {
        "schema_version": 2,
        "models": {
            "TestModel": {
                "hidden_size": 8,
                "dense_intermediate_size": 16,
                "routed_intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_experts": 4,
                "num_shared_experts": 1,
                "top_k": 2,
                "tp_size": 1,
                "ep_size": 1,
                "window_size": 0,
                "weight_dtype": "bf16",
                "activation_dtype": "bf16",
                "kv_cache_dtype": "bf16",
                "attention_type": "auto",
                "distribution": "balanced",
                "a_elementwise": ["rmsnorm", "rope", "residual_add"],
                "f_elementwise": ["rmsnorm", "residual_add"],
                "component_includes": includes,
            }
        },
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


def _generate_package(root):
    source, shapes, observations = _fixture_inputs()
    generator = OperatorDataGenerator(
        dataset_version="test-v1",
        hardware_profiles={
            "a-device": _hardware("a-device"),
            "f-device": _hardware("f-device"),
        },
        sources=[source],
        calibrations={"gemm": Calibration(1.2, 0.1, "gemm-cal-v1")},
    )
    reports = generator.generate(
        root,
        "af-e2e",
        shapes,
        observations,
        holdout_comparisons={
            "gemm": [
                {
                    "shape": "holdout-1",
                    "actual_us": 10.0,
                    "predicted_us": 11.0,
                    "actual_bottleneck": "compute",
                    "predicted_bottleneck": "compute",
                    "target_hardware_ground_truth": True,
                }
            ]
        },
    )
    manifest_path = Path(root) / "model_operators.yaml"
    _write_manifest(manifest_path)
    return reports, manifest_path


def _generate_formal_decode_package(root):
    source, shapes, observations = _fixture_inputs()
    shapes = [
        item
        for item in shapes
        if not (
            item["table"] == "generation_attention"
            and item["key"]["context_len"] == 2
            and item["key"]["num_heads"] == 2
        )
    ]
    communication = [
        replace(
            item,
            key={
                **item.key,
                "effective_bandwidth_bytes_per_s": 10_000.0,
            },
        )
        for item in observations
        if item.table_name == "communication"
    ]
    generator = OperatorDataGenerator(
        dataset_version="formal-decode-v2",
        hardware_profiles={
            "a-device": _hardware("a-device"),
            "f-device": _hardware("f-device"),
        },
        sources=[source],
    )
    generator.generate(
        root,
        "formal-decode-e2e",
        shapes,
        communication,
    )
    manifest_path = Path(root) / "model_operators.yaml"
    _write_manifest(manifest_path)
    return OperatorDataPackage.load(root), manifest_path


def _formal_decode_runtime(cluster):
    moe_config = MoEModelConfig(
        is_moe_model=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        num_shared_experts=1,
        num_moe_layers=1,
        first_k_dense_replace=0,
        intermediate_size=16,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    roofline = AFRooflineStub(layers=1)
    roofline.models["TestModel"].Dmodel = 8
    roofline.models["TestModel"].Nhead = 2
    roofline.models["TestModel"].FFN_Hidden = 16
    runtime_plan = compile_af_runtime_plan(
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        roofline=roofline,
        model="TestModel",
        moe_config=moe_config,
        block_size=2,
        dtype="bf16",
    )
    return moe_config, runtime_plan


def _formal_decode_executor(
    env,
    cluster,
    compute,
    transfer,
    moe_config,
    runtime_plan,
):
    return AFExecutor(
        env=env,
        cluster_config=cluster,
        worker_configs=cluster.workers(),
        compute_provider=compute,
        transfer_provider=transfer,
        model_id="TestModel",
        dtype="bf16",
        total_layers=1,
        hidden_size=8,
        moe_config=moe_config,
        runtime_plan=runtime_plan,
    )


class OperatorDataGenerationTest(unittest.TestCase):
    def test_formula_matches_hand_calculated_gemm_roofline(self):
        key = _key("gemm", m=2, n=4, k=8)
        estimate = estimate_operator(
            "gemm", key, _hardware("a-device"), Calibration(2.0, 1.0, "cal")
        )
        self.assertEqual(estimate.flops, 128.0)
        self.assertEqual(estimate.weight_bytes, 64.0)
        self.assertEqual(estimate.activation_bytes, 48.0)
        raw = max(128 / 1e12 * 1e6, 112 / 1e11 * 1e6)
        self.assertAlmostEqual(estimate.latency_us, raw * 2 + 1)
        self.assertEqual(estimate.calibration_id, "cal")

    def test_mixed_weight_and_activation_dtypes_use_independent_byte_widths(self):
        estimate = estimate_operator(
            "gemm",
            _key(
                "gemm",
                weight_dtype="fp8",
                activation_dtype="bf16",
                m=2,
                n=4,
                k=8,
            ),
            _hardware("a-device"),
        )

        self.assertEqual(estimate.weight_bytes, 8 * 4 * 1)
        self.assertEqual(estimate.activation_bytes, (2 * 8 + 2 * 4) * 2)
        self.assertEqual(estimate.details["formula_inputs"]["weight_element_bytes"], 1)
        self.assertEqual(
            estimate.details["formula_inputs"]["activation_element_bytes"], 2
        )

    def test_formula_covers_prefill_decode_moe_and_elementwise(self):
        hardware = _hardware("a-device")
        prefill = estimate_operator(
            "context_attention", _key("context_attention"), hardware
        )
        decode = estimate_operator(
            "generation_attention", _key("generation_attention"), hardware
        )
        elementwise = estimate_operator("elementwise", _key("elementwise"), hardware)
        moe = estimate_operator("moe", _key("moe"), hardware)
        self.assertEqual(prefill.flops, 128.0)
        self.assertEqual(decode.flops, 64.0)
        self.assertEqual(elementwise.flops, 80.0)
        self.assertEqual(moe.flops, 1536.0)
        self.assertTrue(
            all(
                estimate.latency_us
                >= max(estimate.compute_time_us, estimate.memory_time_us)
                for estimate in (prefill, decode, elementwise, moe)
            )
        )

    def test_conversion_formula_accounts_for_both_dtype_widths(self):
        estimate = estimate_operator(
            "elementwise",
            _key(
                "elementwise",
                op_name="encode",
                activation_dtype="bf16",
                source_dtype="bf16",
                target_dtype="fp8",
                num_tokens=1,
                hidden_size=8,
            ),
            _hardware("a-device"),
        )

        self.assertEqual(estimate.activation_bytes, 8 * (2 + 1))
        self.assertEqual(
            estimate.details["formula_inputs"]["source_element_bytes"], 2
        )
        self.assertEqual(
            estimate.details["formula_inputs"]["target_element_bytes"], 1
        )

    def test_moe_uses_routed_assignments_once_and_tp_shards_compute(self):
        hardware = _hardware("f-device")
        routed = _key(
            "moe",
            device="f-device",
            num_tokens=4,
            top_k=2,
            num_experts=1,
        )
        tp1 = estimate_operator("moe", routed, hardware)
        tp2 = estimate_operator("moe", {**routed, "tp_size": 2}, hardware)
        top_k_one = estimate_operator("moe", {**routed, "top_k": 1}, hardware)

        self.assertEqual(tp1.flops, top_k_one.flops)
        self.assertEqual(tp2.flops, tp1.flops / 2)
        self.assertEqual(tp2.weight_bytes, tp1.weight_bytes / 2)

    def test_moe_swiglu_counts_three_matrices_and_rejects_uneven_tp(self):
        key = _key(
            "moe",
            device="f-device",
            operation="shared_expert",
            num_tokens=2,
            num_experts=1,
            top_k=1,
        )
        estimate = estimate_operator("moe", key, _hardware("f-device"))

        self.assertEqual(estimate.flops, 6 * 2 * 8 * 16)
        self.assertEqual(estimate.weight_bytes, 3 * 1 * 8 * 16 * 2)
        self.assertEqual(estimate.activation_bytes, (2 * 8 + 4 * 16 + 2 * 8) * 2)
        self.assertEqual(estimate.details["formula_inputs"]["projection_count"], 3)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            estimate_operator(
                "moe",
                {**key, "inter_size": 15, "tp_size": 2},
                _hardware("f-device"),
            )

    def test_attention_uses_gqa_mqa_widths_and_caps_decode_window(self):
        common = _key(
            "generation_attention",
            attention_type="gqa",
            context_len=9,
            window_size=4,
            num_heads=8,
            num_kv_heads=2,
            head_dim=4,
        )
        gqa = estimate_operator(
            "generation_attention", common, _hardware("a-device")
        )
        mqa = estimate_operator(
            "generation_attention",
            {**common, "attention_type": "mqa", "num_kv_heads": 1},
            _hardware("a-device"),
        )
        mha = estimate_operator(
            "generation_attention",
            {**common, "attention_type": "mha", "num_kv_heads": 8},
            _hardware("a-device"),
        )

        self.assertEqual(gqa.flops, 4 * 1 * 4 * (8 * 4))
        self.assertEqual(gqa.kv_bytes, 2 * 1 * 4 * (2 * 4) * 2)
        self.assertEqual(mqa.kv_bytes, 2 * 1 * 4 * (1 * 4) * 2)
        self.assertEqual(mha.kv_bytes, 2 * 1 * 4 * (8 * 4) * 2)
        self.assertEqual(
            gqa.details["formula_inputs"]["effective_context_length"], 4
        )

    def test_mla_and_mfa_require_dedicated_attention_formulas(self):
        for attention_type in ("mla", "mfa"):
            with self.subTest(attention_type=attention_type):
                with self.assertRaisesRegex(ValueError, "dedicated formula"):
                    estimate_operator(
                        "generation_attention",
                        _key(
                            "generation_attention",
                            attention_type=attention_type,
                        ),
                        _hardware("a-device"),
                    )

    def test_dense_ffn_operator_keys_use_tp_local_intermediate_shape(self):
        class RecordingLookup:
            def __init__(self):
                self.calls = []
                self.package = SimpleNamespace(dataset_version="shape-test")

            def lookup(self, table_name, key):
                self.calls.append((table_name, dict(key)))
                return LookupResult(0.0, "exact", "shape-test", dict(key))

        lookup = RecordingLookup()
        manifest = ModelOperatorManifest(
            {
                "DenseModel": ModelShape(
                    model_id="DenseModel",
                    hidden_size=8,
                    dense_intermediate_size=16,
                    routed_intermediate_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    f_elementwise=(),
                )
            }
        )
        provider = OperatorDataComputeLatencyProvider(lookup, manifest, "analytical")
        query = AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            stage="expert",
            token_count=2,
            context_lengths=(2,),
            model_id="DenseModel",
            dtype="bf16",
            device_type="f-device",
            worker_id=0,
            operation="expert_rank_compute",
            source_domain="f-domain",
            local_shape={
                "tokens": 2,
                "hidden_size": 8,
                "global_hidden_size": 8,
                "tp_size": 2,
                "intermediate_size": 8,
            },
            local_token_count=2,
            tp_group=(0, 1),
        )

        provider.estimate(query)
        gemms = [key for table, key in lookup.calls if table == "gemm"]
        swiglu = next(
            key
            for table, key in lookup.calls
            if table == "elementwise" and key["op_name"] == "swiglu"
        )
        self.assertEqual(gemms[0]["n"], 16)
        self.assertEqual(gemms[1]["k"], 8)
        self.assertEqual(swiglu["hidden_size"], 8)

    def test_decode_attention_queries_whole_ragged_batch_and_uses_gqa_local_widths(self):
        class RecordingLookup:
            def __init__(self):
                self.calls = []
                self.package = SimpleNamespace(dataset_version="shape-test")

            def lookup(self, table_name, key):
                self.calls.append((table_name, dict(key)))
                latency = 3.0 if table_name == "generation_attention_batch" else 0.0
                return LookupResult(latency, "exact", "shape-test", dict(key))

        lookup = RecordingLookup()
        model = ModelShape(
            model_id="GQAModel",
            hidden_size=32,
            dense_intermediate_size=64,
            routed_intermediate_size=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            tp_size=2,
            window_size=4,
            attention_type="gqa",
            a_elementwise=(),
        )
        provider = OperatorDataComputeLatencyProvider(
            lookup, ModelOperatorManifest({"GQAModel": model}), "analytical"
        )
        query = AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            stage="a",
            token_count=2,
            context_lengths=(1, 9),
            model_id="GQAModel",
            dtype="bf16",
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="bf16",
            device_type="a-device",
            worker_id=0,
            operation="a_rank_compute",
            source_domain="a-domain",
            local_shape={
                "tokens": 2,
                "hidden_size": 16,
                "global_hidden_size": 32,
                "tp_size": 2,
                "num_heads": 4,
                "num_kv_heads": 1,
                "head_dim": 4,
                "query_width": 16,
                "kv_width": 4,
            },
            local_token_count=2,
            tp_group=(0, 1),
        )

        estimate = provider.estimate(query)
        attention = next(
            item
            for item in estimate.details["components"]
            if item["name"] == "attention_core"
        )
        generation_keys = [
            key for table, key in lookup.calls if table == "generation_attention_batch"
        ]
        gemm_keys = [key for table, key in lookup.calls if table == "gemm"]

        self.assertEqual([key["context_lengths"] for key in generation_keys], ["1,4"])
        self.assertTrue(all(key["batch_size"] == 2 for key in generation_keys))
        self.assertTrue(all(key["num_heads"] == 4 for key in generation_keys))
        self.assertTrue(all(key["num_kv_heads"] == 1 for key in generation_keys))
        self.assertEqual(gemm_keys[0]["n"], 24)
        self.assertEqual(gemm_keys[0]["weight_dtype"], "fp8")
        self.assertEqual(gemm_keys[0]["activation_dtype"], "bf16")
        self.assertEqual(gemm_keys[1]["k"], 16)
        self.assertEqual(attention["latency_us"], 3.0)
        self.assertEqual(
            [item["context_len"] for item in attention["per_request"]], [1, 9]
        )

    def test_provider_rejects_implicit_uneven_attention_tp_shards(self):
        class NeverLookup:
            package = SimpleNamespace(dataset_version="shape-test")

            def lookup(self, table_name, key):
                raise AssertionError("invalid TP shape must fail before lookup")

        model = ModelShape(
            model_id="UnevenModel",
            hidden_size=24,
            dense_intermediate_size=48,
            routed_intermediate_size=48,
            num_attention_heads=8,
            num_key_value_heads=2,
            tp_size=3,
            attention_type="gqa",
            a_elementwise=(),
        )
        provider = OperatorDataComputeLatencyProvider(
            NeverLookup(), ModelOperatorManifest({"UnevenModel": model}), "analytical"
        )
        query = AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            stage="a",
            token_count=1,
            context_lengths=(1,),
            model_id="UnevenModel",
            dtype="bf16",
            device_type="a-device",
            worker_id=0,
            operation="a_rank_compute",
            local_shape={
                "tokens": 1,
                "hidden_size": 8,
                "global_hidden_size": 24,
                "tp_size": 3,
                "num_heads": 3,
                "num_kv_heads": 1,
                "head_dim": 3,
                "query_width": 9,
                "kv_width": 3,
            },
            local_token_count=1,
            tp_group=(0, 1, 2),
        )

        with self.assertRaisesRegex(ValueError, "query heads.*not divisible"):
            provider.estimate(query)

    def test_attention_runtime_shape_must_match_manifest_before_lookup(self):
        class NeverLookup:
            package = SimpleNamespace(dataset_version="shape-test")

            def lookup(self, table_name, key):
                raise AssertionError("shape mismatch must fail before lookup")

        model = ModelShape(
            model_id="ShapeCheckedModel",
            hidden_size=32,
            dense_intermediate_size=64,
            routed_intermediate_size=32,
            num_attention_heads=8,
            num_key_value_heads=2,
            tp_size=2,
            attention_type="gqa",
            a_elementwise=(),
        )
        provider = OperatorDataComputeLatencyProvider(
            NeverLookup(),
            ModelOperatorManifest({"ShapeCheckedModel": model}),
            "analytical",
        )
        query = AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            stage="attention",
            token_count=1,
            context_lengths=(1,),
            model_id="ShapeCheckedModel",
            dtype="bf16",
            device_type="a-device",
            worker_id=0,
            operation="attention",
            source_domain="a-domain",
            local_shape={
                "tokens": 1,
                "hidden_size": 32,
                "global_hidden_size": 32,
                "tp_size": 2,
                "num_heads": 3,
                "num_kv_heads": 1,
                "head_dim": 4,
                "query_width": 12,
                "kv_width": 4,
            },
            local_token_count=1,
            original_token_count=1,
            output_token_count=1,
            layer_type="dense",
            tp_group=(0, 1),
        )

        with self.assertRaisesRegex(
            ValueError, "Attention local shape does not match operator manifest"
        ):
            provider.estimate(query)

    def test_dense_routed_and_shared_intermediate_shapes_are_distinct(self):
        class RecordingLookup:
            def __init__(self):
                self.calls = []
                self.package = SimpleNamespace(dataset_version="shape-test")

            def lookup(self, table_name, key):
                self.calls.append((table_name, dict(key)))
                return LookupResult(0.0, "exact", "shape-test", dict(key))

        lookup = RecordingLookup()
        model = ModelShape(
            model_id="DistinctFFNModel",
            hidden_size=8,
            dense_intermediate_size=64,
            routed_intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            num_shared_experts=2,
            top_k=2,
            f_elementwise=(),
        )
        provider = OperatorDataComputeLatencyProvider(
            lookup,
            ModelOperatorManifest({"DistinctFFNModel": model}),
            "analytical",
        )

        def estimate(stage, intermediate_size, **overrides):
            values = {
                "iteration_id": 0,
                "block_id": 0,
                "layer_id": 0,
                "phase": "decode",
                "stage": stage,
                "token_count": 2,
                "context_lengths": (1, 1),
                "model_id": "DistinctFFNModel",
                "dtype": "bf16",
                "device_type": "f-device",
                "worker_id": 0,
                "operation": stage,
                "source_domain": "f-domain",
                "local_shape": {
                    "tokens": 2,
                    "hidden_size": 8,
                    "global_hidden_size": 8,
                    "tp_size": 1,
                    "intermediate_size": intermediate_size,
                },
                "local_token_count": 2,
                "original_token_count": 2,
                "output_token_count": 2,
                "layer_type": "dense" if stage == "dense_ffn" else "moe",
                "tp_group": (0,),
            }
            values.update(overrides)
            return provider.estimate(AFComputeQuery(**values))

        estimate("dense_ffn", 64)
        estimate(
            "routed_expert",
            16,
            assignment_count=2,
            expert_ids=(0,),
            expert_workload=((0, 2),),
        )
        estimate("shared_expert", 32)

        with self.assertRaisesRegex(
            ValueError, "local intermediate_size=16 does not match expected=64"
        ):
            estimate("dense_ffn", 16)
        with self.assertRaisesRegex(
            ValueError, "local intermediate_size=64 does not match expected=16"
        ):
            estimate(
                "routed_expert",
                64,
                assignment_count=2,
                expert_ids=(0,),
                expert_workload=((0, 2),),
            )
        with self.assertRaisesRegex(
            ValueError, "local intermediate_size=16 does not match expected=32"
        ):
            estimate("shared_expert", 16)

    def test_rank_reuse_expert_uses_expert_tp_shape_not_base_f_tp(self):
        class RecordingLookup:
            def __init__(self):
                self.calls = []
                self.package = SimpleNamespace(dataset_version="shape-test")

            def lookup(self, table_name, key):
                self.calls.append((table_name, dict(key)))
                return LookupResult(0.0, "exact", "shape-test", dict(key))

        lookup = RecordingLookup()
        model = ModelShape(
            model_id="RankReuseModel",
            hidden_size=8,
            dense_intermediate_size=16,
            routed_intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            top_k=2,
            f_elementwise=(),
        )
        provider = OperatorDataComputeLatencyProvider(
            lookup,
            ModelOperatorManifest({"RankReuseModel": model}),
            "analytical",
        )
        query = AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            stage="expert",
            token_count=1,
            context_lengths=(1,),
            model_id="RankReuseModel",
            dtype="bf16",
            device_type="f-device",
            worker_id=0,
            operation="expert_rank_compute",
            source_domain="f-domain",
            local_shape={
                "tokens": 1,
                "hidden_size": 8,
                "global_hidden_size": 8,
                "tp_size": 2,
                "expert_tp_size": 1,
                "intermediate_size": 16,
            },
            local_token_count=1,
            expert_ids=(0,),
            expert_workload=((0, 1),),
            tp_group=(0, 1),
            ep_group=(0, 1, 2, 3),
        )

        provider.estimate(query)
        moe_key = next(key for table, key in lookup.calls if table == "moe")
        self.assertEqual(moe_key["tp_size"], 1)
        self.assertEqual(moe_key["hidden_size"], 8)

    def test_formal_compute_operations_have_disjoint_compositions(self):
        class RecordingLookup:
            def __init__(self):
                self.calls = []
                self.package = SimpleNamespace(dataset_version="shape-test")

            def lookup(self, table_name, key):
                self.calls.append((table_name, dict(key)))
                return LookupResult(1.0, "exact", "shape-test", dict(key))

        lookup = RecordingLookup()
        model = ModelShape(
            model_id="FormalModel",
            hidden_size=8,
            dense_intermediate_size=16,
            routed_intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_experts=4,
            num_shared_experts=2,
            top_k=2,
            a_elementwise=("rmsnorm", "rope", "residual_add", "data_reorder"),
            f_elementwise=("rmsnorm", "residual_add", "data_reorder"),
        )
        provider = OperatorDataComputeLatencyProvider(
            lookup,
            ModelOperatorManifest({"FormalModel": model}),
            "analytical",
        )

        def query(stage, layer_type, shape, **overrides):
            tokens = overrides.pop("tokens", 2)
            values = {
                "iteration_id": 0,
                "block_id": 0,
                "layer_id": 0,
                "phase": "decode",
                "stage": stage,
                "token_count": tokens,
                "context_lengths": (1, 2),
                "model_id": "FormalModel",
                "dtype": "bf16",
                "weight_dtype": "bf16",
                "activation_dtype": "bf16",
                "kv_cache_dtype": "bf16",
                "device_type": (
                    "f-device"
                    if stage in {"dense_ffn", "routed_expert", "shared_expert"}
                    else "a-device"
                ),
                "worker_id": 0,
                "operation": "router_topk" if stage == "router" else stage,
                "source_domain": "f-domain" if stage.endswith("expert") or stage == "dense_ffn" else "a-domain",
                "local_shape": {"tokens": tokens, **shape},
                "local_token_count": tokens,
                "layer_type": layer_type,
                "original_token_count": tokens,
                "output_token_count": tokens,
            }
            values.update(overrides)
            return AFComputeQuery(**values)

        attention = provider.estimate(
            query(
                "attention",
                "dense",
                {
                    "hidden_size": 8,
                    "global_hidden_size": 8,
                    "tp_size": 1,
                    "num_heads": 2,
                    "num_kv_heads": 2,
                    "head_dim": 4,
                    "query_width": 8,
                    "kv_width": 8,
                },
            )
        )
        dense = provider.estimate(
            query(
                "dense_ffn",
                "dense",
                {
                    "hidden_size": 8,
                    "global_hidden_size": 8,
                    "tp_size": 1,
                    "intermediate_size": 16,
                },
            )
        )
        router = provider.estimate(
            query(
                "router",
                "moe",
                {"hidden_size": 8, "num_experts": 4, "top_k": 2},
                assignment_count=4,
            )
        )
        routed = provider.estimate(
            query(
                "routed_expert",
                "moe",
                {
                    "hidden_size": 8,
                    "global_hidden_size": 8,
                    "tp_size": 1,
                    "expert_tp_size": 1,
                    "intermediate_size": 16,
                },
                tokens=4,
                original_token_count=2,
                assignment_count=4,
                output_token_count=4,
                expert_ids=(0, 3),
                expert_workload=((0, 1), (3, 3)),
                routing_edges=(),
            )
        )
        shared = provider.estimate(
            query(
                "shared_expert",
                "moe",
                {
                    "hidden_size": 8,
                    "global_hidden_size": 8,
                    "tp_size": 1,
                    "intermediate_size": 32,
                },
            )
        )
        residual = provider.estimate(
            query("residual", "moe", {"hidden_size": 8})
        )
        layout = provider.estimate(query("layout", "moe", {"hidden_size": 8}))

        edges = tuple(
            AFRoutingEdge(
                iteration_id=0,
                phase="decode",
                decode_step=0,
                layer_id=0,
                request_id=request_id,
                token_id=0,
                top_k_index=top_k_index,
                expert_id=2 * request_id + top_k_index,
                routing_weight=weight,
                a_dp_rank=0,
                a_tp_rank=0,
            )
            for request_id in range(2)
            for top_k_index, weight in ((0, 0.75), (1, 0.25))
        )
        routed_slices = tuple(
            AFTensorSlice(
                tensor_item_id=f"routed:{edge.request_id}:{edge.top_k_index}",
                interval=(0, 8),
                layout="partial",
                partial_id=f"expert:{edge.expert_id}:k{edge.top_k_index}",
                routing_weight=edge.routing_weight,
                element_count=8,
                payload_bytes=16,
            )
            for edge in edges
        )
        shared_slices = tuple(
            AFTensorSlice(
                tensor_item_id=f"shared:{request_id}",
                interval=(0, 8),
                layout="partial",
                partial_id="shared",
                element_count=8,
                payload_bytes=16,
            )
            for request_id in range(2)
        )
        combine = provider.estimate(
            query(
                "weighted_combine",
                "moe",
                {"hidden_size": 8},
                assignment_count=4,
                input_edge_count=6,
                routing_edges=edges,
                tensor_slices=routed_slices + shared_slices,
            )
        )
        encoded = provider.estimate(
            query(
                "encode",
                "moe",
                {"elements": 8},
                tokens=1,
                original_token_count=1,
                output_token_count=1,
                source_dtype="bf16",
                target_dtype="fp8",
            )
        )
        decoded = provider.estimate(
            query(
                "decode",
                "moe",
                {"elements": 8},
                tokens=1,
                original_token_count=1,
                output_token_count=1,
                source_dtype="fp8",
                target_dtype="bf16",
            )
        )

        names = lambda estimate: [
            item["name"] for item in estimate.details["components"]
        ]
        self.assertNotIn("a_residual_add", names(attention))
        self.assertEqual(
            names(dense),
            ["f_rmsnorm", "dense_gate_up", "dense_swiglu", "dense_down"],
        )
        self.assertEqual(names(router), ["f_rmsnorm", "router"])
        self.assertEqual(names(routed), ["routed_expert"])
        self.assertEqual(names(shared), ["shared_expert"])
        self.assertEqual(names(residual), ["a_residual_add"])
        self.assertEqual(names(layout), ["a_data_reorder"])
        self.assertEqual(
            names(combine),
            ["routing_weight_multiply", "routed_reduce", "shared_sum"],
        )
        self.assertEqual(
            [item["routing_weight"] for item in combine.details["routing_edges"]],
            [0.75, 0.25, 0.75, 0.25],
        )
        shared_key = shared.details["components"][0]["key"]
        self.assertEqual(shared_key["num_tokens"], 2)
        self.assertEqual(shared_key["inter_size"], 32)
        conversion_key = encoded.details["components"][0]["key"]
        self.assertEqual(conversion_key["source_dtype"], "bf16")
        self.assertEqual(conversion_key["target_dtype"], "fp8")
        decode_key = decoded.details["components"][0]["key"]
        self.assertEqual(decode_key["source_dtype"], "fp8")
        self.assertEqual(decode_key["target_dtype"], "bf16")
        self.assertFalse(any(table == "communication" for table, _ in lookup.calls))

    def test_collective_provider_derives_wire_volume_once_from_logical_bytes(self):
        class RecordingLookup:
            package = SimpleNamespace(dataset_version="shape-test")

            def __init__(self):
                self.keys = []

            def link_profile(self, key):
                self.keys.append(dict(key))
                return LookupResult(
                    0.0,
                    "exact",
                    "shape-test",
                    dict(key),
                    values={
                        "fixed_latency_us": 0.0,
                        "effective_bandwidth_bytes_per_s": 100.0,
                    },
                )

        lookup = RecordingLookup()
        provider = OperatorDataTransferLatencyProvider(
            lookup, "analytical", "collective-fabric", 4
        )
        base = AFTransferQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            direction="all_reduce",
            token_count=1,
            hidden_size=50,
            dtype="bf16",
            activation_dtype="bf16",
            source_device="a-device",
            target_device="a-device",
            network="tp-net",
            operation="all_reduce",
            group_size=4,
            payload_bytes=None,
            source_domain="a-domain",
            target_domain="a-domain",
            participant_ranks=(0, 1, 2, 3),
            resource_group="collective-fabric",
            resource_lane="shared",
            logical_input_bytes_per_rank=((0, 100), (1, 100), (2, 100), (3, 100)),
            logical_output_bytes_per_rank=((0, 100), (1, 100), (2, 100), (3, 100)),
        )

        all_reduce = provider.estimate(base)
        all_gather = provider.estimate(
            replace(
                base,
                direction="all_gather",
                operation="all_gather",
                logical_input_bytes_per_rank=((0, 25), (1, 25), (2, 25), (3, 25)),
                logical_output_bytes_per_rank=((0, 100), (1, 100), (2, 100), (3, 100)),
            )
        )
        reduce_scatter = provider.estimate(
            replace(
                base,
                direction="reduce_scatter",
                operation="reduce_scatter",
                logical_input_bytes_per_rank=((0, 100), (1, 100), (2, 100), (3, 100)),
                logical_output_bytes_per_rank=((0, 25), (1, 25), (2, 25), (3, 25)),
            )
        )

        self.assertEqual(all_reduce.payload_bytes, 150)
        self.assertEqual(all_gather.payload_bytes, 75)
        self.assertEqual(reduce_scatter.payload_bytes, 75)
        self.assertEqual(
            [key["direction"] for key in lookup.keys],
            ["collective", "collective", "collective"],
        )
        self.assertEqual(all_reduce.details["planner_payload_bytes"], None)
        self.assertEqual(
            all_reduce.details["logical_input_bytes_per_rank"],
            ((0, 100), (1, 100), (2, 100), (3, 100)),
        )
        self.assertNotIn("queue_duration", all_reduce.details)

    def test_endpoint_leg_provider_preserves_planned_payload_and_identity(self):
        class RecordingLookup:
            package = SimpleNamespace(dataset_version="shape-test")

            def __init__(self):
                self.key = None

            def link_profile(self, key):
                self.key = dict(key)
                return LookupResult(
                    0.0,
                    "exact",
                    "shape-test",
                    dict(key),
                    values={
                        "fixed_latency_us": 1.0,
                        "effective_bandwidth_bytes_per_s": 8_000_000.0,
                    },
                )

        source = AFEndpoint("a-domain", 0, "a-pool:0", "a-device")
        target = AFEndpoint("f-domain", 1, "f-pool:1", "f-device", tp_rank=1)
        tensor_slice = AFTensorSlice(
            tensor_item_id="request:7:token:3:routed:k0:expert:2",
            interval=(0, 8),
            layout="sharded",
            owner_domain="a-domain",
            owner_domain_rank=0,
            partial_id="expert:2:k0",
            routing_weight=0.6,
            element_count=8,
            payload_bytes=8,
        )
        edge = AFRoutingEdge(
            iteration_id=0,
            phase="decode",
            decode_step=3,
            layer_id=1,
            request_id=7,
            token_id=3,
            top_k_index=0,
            expert_id=2,
            routing_weight=0.6,
            a_dp_rank=0,
            a_tp_rank=0,
        )
        lookup = RecordingLookup()
        provider = OperatorDataTransferLatencyProvider(lookup, "analytical")
        query = AFTransferQuery(
            iteration_id=0,
            block_id=0,
            layer_id=1,
            phase="decode",
            direction="a_to_f",
            token_count=1,
            hidden_size=8,
            dtype="bf16",
            activation_dtype="bf16",
            a_to_f_dtype="fp8",
            source_device="a-device",
            target_device="f-device",
            network="af-net",
            operation="af_a_to_f",
            group_size=2,
            payload_bytes=8,
            communication_kind="endpoint_leg",
            logical_stage_id="layer:1:a_to_f",
            leg_id="layer:1:a_to_f:leg:0",
            source_endpoint=source,
            target_endpoint=target,
            participant_endpoints=(source, target),
            tensor_slices=(tensor_slice,),
            source_layout="sharded",
            target_layout="sharded",
            resource_group="af-fabric",
            resource_lane="a_to_f",
            routing_edges=(edge,),
        )

        estimate = provider.estimate(query)
        self.assertEqual(estimate.payload_bytes, 8)
        self.assertEqual(lookup.key["transfer_dtype"], "fp8")
        self.assertEqual(estimate.details["transfer_element_bytes"], 1)
        self.assertEqual(estimate.details["logical_stage_id"], "layer:1:a_to_f")
        self.assertEqual(
            estimate.details["tensor_slices"][0]["tensor_item_id"],
            tensor_slice.tensor_item_id,
        )
        self.assertEqual(
            estimate.details["routing_edges"][0]["routing_weight"], 0.6
        )
        with self.assertRaisesRegex(ValueError, "tensor slice payload sum"):
            provider.estimate(replace(query, payload_bytes=16))

    def test_source_grades_cannot_masquerade_as_measurements(self):
        with self.assertRaisesRegex(ValueError, "grade A"):
            SourceRecord(
                "bad",
                EvidenceGrade.C,
                "measured",
                "groq",
                "backend",
                "formula",
                "hash",
            )
        with self.assertRaisesRegex(ValueError, "grade B"):
            SourceRecord(
                "bad-public",
                EvidenceGrade.A,
                "public_benchmark",
                "groq",
                "backend",
                "endpoint",
                "hash",
            )

    def test_measured_rows_override_formula_and_bundle_is_reproducible(self):
        with _tempdir() as first, _tempdir() as second:
            source, shapes, observations = _fixture_inputs()
            generator = OperatorDataGenerator(
                "test-v1",
                {"a-device": _hardware("a-device"), "f-device": _hardware("f-device")},
                [source],
            )
            generator.generate(first, "af-e2e", shapes, observations)
            generator.generate(
                second, "af-e2e", list(reversed(shapes)), list(reversed(observations))
            )
            first_package = OperatorDataPackage.load(first)
            second_package = OperatorDataPackage.load(second)
            query = _key("gemm", m=2, n=24, k=8)
            result = OperatorDataLookup(first_package).lookup("gemm", query)
            self.assertEqual(result.latency_us, 20.0)
            self.assertEqual(result.source_id, source.source_id)
            self.assertEqual(
                first_package.metadata["input_hash"],
                second_package.metadata["input_hash"],
            )
            self.assertEqual(first_package.tables, second_package.tables)

    def test_generated_analysis_rows_preserve_versioned_formula_provenance(self):
        with _tempdir() as root:
            _generate_package(root)
            package = OperatorDataPackage.load(root)

            row = package.analysis_tables["gemm"][0]
            self.assertEqual(
                row["details"]["formula_version"], "tokensim-analytical-v2"
            )
            self.assertIn("formula", row["details"])
            self.assertIn("formula_inputs", row["details"])

    def test_holdout_report_has_required_metrics_and_claim_boundary(self):
        report = ValidationReport.from_holdout(
            2,
            [
                {
                    "shape": "one",
                    "actual_us": 10.0,
                    "predicted_us": 12.0,
                    "actual_bottleneck": "memory",
                    "predicted_bottleneck": "memory",
                }
            ],
            has_target_ground_truth=False,
        )
        self.assertEqual(report.coverage, 0.5)
        self.assertAlmostEqual(report.mean_relative_error, 0.2)
        self.assertAlmostEqual(report.p95_relative_error, 0.2)
        self.assertEqual(report.bottleneck_consistency, 1.0)
        self.assertIn("self-consistency", report.claim)

    def test_parquet_reader_reports_missing_dependency_at_entry(self):
        with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
            with self.assertRaisesRegex(
                OperatorDataValidationError, "requires pyarrow"
            ):
                _read_parquet(Path("unused.parquet"))


class OperatorDataLookupContractTest(unittest.TestCase):
    @staticmethod
    def _lookup(table_name, rows):
        package = SimpleNamespace(
            dataset_version="memory-v2",
            tables={table_name: tuple(rows)},
        )
        return OperatorDataLookup(package)

    @staticmethod
    def _row(key, latency_us):
        return {**key, "latency_us": latency_us, "source_id": "memory:v2"}

    def test_gemm_interpolates_only_m_with_identical_discrete_keys(self):
        lookup = self._lookup(
            "gemm",
            (
                self._row(_key("gemm", m=1, n=7, k=8), 10.0),
                self._row(_key("gemm", m=3, n=7, k=8), 30.0),
            ),
        )

        result = lookup.lookup("gemm", _key("gemm", m=2, n=7, k=8))
        self.assertEqual(result.match_type, "interpolated")
        self.assertEqual(result.latency_us, 20.0)
        for invalid in (
            _key("gemm", weight_dtype="fp8", m=2, n=7, k=8),
            _key("gemm", activation_dtype="fp8", m=2, n=7, k=8),
            _key("gemm", m=2, n=9, k=8),
            _key("gemm", m=2, n=7, k=9),
            _key("gemm", m=4, n=7, k=8),
        ):
            with self.subTest(key=invalid):
                with self.assertRaises(MissingOperatorDataError):
                    lookup.lookup("gemm", invalid)

    def test_routed_and_shared_expert_rows_never_form_interpolation_pair(self):
        lookup = self._lookup(
            "moe",
            (
                self._row(
                    _key(
                        "moe",
                        device="f-device",
                        operation="routed_expert",
                        num_tokens=1,
                        top_k=1,
                        num_experts=1,
                    ),
                    10.0,
                ),
                self._row(
                    _key(
                        "moe",
                        device="f-device",
                        operation="shared_expert",
                        num_tokens=3,
                        top_k=1,
                        num_experts=1,
                    ),
                    30.0,
                ),
            ),
        )

        with self.assertRaisesRegex(
            MissingOperatorDataError, r"available values=\[1\]"
        ):
            lookup.lookup(
                "moe",
                _key(
                    "moe",
                    device="f-device",
                    operation="routed_expert",
                    num_tokens=2,
                    top_k=1,
                    num_experts=1,
                ),
            )

    def test_manifest_validates_but_does_not_override_effective_dtypes(self):
        manifest = ModelOperatorManifest(
            {
                "MixedModel": ModelShape(
                    model_id="MixedModel",
                    hidden_size=8,
                    dense_intermediate_size=16,
                    routed_intermediate_size=16,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    weight_dtype="fp8",
                    activation_dtype="bf16",
                    kv_cache_dtype="fp16",
                )
            }
        )
        matching = AFEffectiveDTypes(
            weight_dtype="fp8",
            activation_dtype="bf16",
            kv_cache_dtype="fp16",
            a_to_f_dtype="fp8",
            f_to_a_dtype="bf16",
            sources=(),
        )
        manifest.validate_effective_dtypes("MixedModel", matching)

        mismatched = replace(matching, activation_dtype="fp16")
        with self.assertRaisesRegex(
            ValueError,
            "model='MixedModel'.*role='activation_dtype'.*effective='fp16'.*"
            "manifest='bf16'",
        ):
            manifest.validate_effective_dtypes("MixedModel", mismatched)
        self.assertEqual(manifest.model("MixedModel").activation_dtype, "bf16")

    def test_collective_provider_derives_wire_bytes_exactly_once(self):
        class RecordingLookup:
            package = SimpleNamespace(dataset_version="memory-v2")

            def __init__(self):
                self.keys = []

            def link_profile(self, key):
                self.keys.append(dict(key))
                return LookupResult(
                    0.0,
                    "exact",
                    "memory:v2",
                    dict(key),
                    values={
                        "fixed_latency_us": 0.0,
                        "effective_bandwidth_bytes_per_s": 1e9,
                    },
                )

        lookup = RecordingLookup()
        provider = OperatorDataTransferLatencyProvider(
            lookup, "analytical", "collective-fabric", 4
        )
        base = AFTransferQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            direction="tp_collective",
            token_count=1,
            hidden_size=8,
            dtype="bf16",
            activation_dtype="bf16",
            source_device="a-device",
            target_device="a-device",
            network="tp-net",
            operation="all_reduce",
            group_size=4,
            payload_bytes=999,
            participant_ranks=(0, 1, 2, 3),
            resource_group="collective-fabric",
            resource_lane="collective",
            logical_input_bytes_per_rank=((0, 80), (1, 80), (2, 80), (3, 80)),
            logical_output_bytes_per_rank=((0, 80), (1, 80), (2, 80), (3, 80)),
        )

        all_reduce = provider.estimate(base)
        all_gather = provider.estimate(
            replace(
                base,
                operation="all_gather",
                logical_input_bytes_per_rank=(
                    (0, 20),
                    (1, 20),
                    (2, 20),
                    (3, 20),
                ),
            )
        )
        reduce_scatter = provider.estimate(
            replace(
                base,
                operation="reduce_scatter",
                logical_output_bytes_per_rank=(
                    (0, 20),
                    (1, 20),
                    (2, 20),
                    (3, 20),
                ),
            )
        )

        self.assertEqual(all_reduce.payload_bytes, 120)
        self.assertEqual(all_gather.payload_bytes, 60)
        self.assertEqual(reduce_scatter.payload_bytes, 60)
        self.assertEqual(all_reduce.details["planner_payload_bytes"], 999)
        self.assertEqual(all_reduce.details["wire_volume_factor"], 1.5)
        self.assertTrue(all(key["direction"] == "collective" for key in lookup.keys))
        self.assertEqual(
            [key["operation"] for key in lookup.keys],
            ["all_reduce", "all_gather", "reduce_scatter"],
        )


class OperatorDataLookupTest(unittest.TestCase):
    def setUp(self):
        self.temp = _tempdir()
        _generate_package(self.temp.name)
        self.package = OperatorDataPackage.load(self.temp.name)
        self.lookup = OperatorDataLookup(self.package)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_is_preferred_and_units_remain_microseconds(self):
        result = self.lookup.lookup("gemm", _key("gemm", m=2, n=24, k=8))
        self.assertEqual(result.match_type, "exact")
        self.assertEqual(result.latency_us, 20.0)
        self.assertEqual(result.source_id, "measured:test")

    def test_each_supported_axis_interpolates_inside_bounds(self):
        cases = [
            ("gemm", _key("gemm", m=2, n=7, k=8), 20.0, "m"),
            (
                "context_attention",
                _key("context_attention", input_seq_len=2),
                10.0,
                "input_seq_len",
            ),
            (
                "generation_attention",
                _key("generation_attention", context_len=2),
                10.0,
                "context_len",
            ),
        ]
        # Attention has an exact midpoint in the common fixture; remove it from a local package index.
        for table, key, expected, axis in cases:
            if table != "gemm":
                spec_rows = tuple(
                    row for row in self.package.tables[table] if row[axis] != 2
                )
                package = self.package.__class__(
                    **{
                        **self.package.__dict__,
                        "tables": {**self.package.tables, table: spec_rows},
                    }
                )
                lookup = OperatorDataLookup(package)
            else:
                lookup = self.lookup
            result = lookup.lookup(table, key)
            self.assertEqual(result.match_type, "interpolated")
            self.assertEqual(result.latency_us, expected)
            self.assertEqual(result.interpolation["axis"], axis)
            self.assertEqual(result.interpolation["left_weight"], 0.5)

        moe_rows = tuple(
            row
            for row in self.package.tables["moe"]
            if not (row["phase"] == "decode" and row["num_tokens"] == 2)
        )
        package = self.package.__class__(
            **{
                **self.package.__dict__,
                "tables": {**self.package.tables, "moe": moe_rows},
            }
        )
        result = OperatorDataLookup(package).lookup(
            "moe", _key("moe", device="f-device", num_tokens=2)
        )
        self.assertEqual(result.match_type, "interpolated")
        self.assertEqual(result.latency_us, 12.0)

    def test_out_of_range_and_elementwise_miss_fail_with_full_key(self):
        with self.assertRaisesRegex(
            MissingOperatorDataError, "available values"
        ) as error:
            self.lookup.lookup("gemm", _key("gemm", m=4, n=7, k=8))
        self.assertEqual(error.exception.key["m"], 4)
        with self.assertRaisesRegex(
            MissingOperatorDataError, "interpolation is disabled"
        ):
            self.lookup.lookup("elementwise", _key("elementwise", num_tokens=99))

    def test_lookup_never_interpolates_across_operation_dtype_or_gemm_shape(self):
        routed_one_shared_three = tuple(
            row
            for row in self.package.tables["moe"]
            if (
                row["phase"] == "decode"
                and row["tp_size"] == 1
                and row["ep_size"] == 1
                and row["num_experts"] == 1
                and row["top_k"] == 1
                and (
                    (row["operation"] == "routed_expert" and row["num_tokens"] == 1)
                    or (
                        row["operation"] == "shared_expert"
                        and row["num_tokens"] == 3
                    )
                )
            )
        )
        package = self.package.__class__(
            **{
                **self.package.__dict__,
                "tables": {**self.package.tables, "moe": routed_one_shared_three},
            }
        )
        with self.assertRaisesRegex(
            MissingOperatorDataError, r"available values=\[1\]"
        ) as operation_error:
            OperatorDataLookup(package).lookup(
                "moe",
                _key(
                    "moe",
                    device="f-device",
                    operation="routed_expert",
                    num_tokens=2,
                    top_k=1,
                    num_experts=1,
                ),
            )
        self.assertEqual(operation_error.exception.key["operation"], "routed_expert")

        with self.assertRaisesRegex(
            MissingOperatorDataError, r"available values=\[\]"
        ):
            self.lookup.lookup(
                "gemm", _key("gemm", weight_dtype="fp8", m=2, n=7, k=8)
            )
        with self.assertRaisesRegex(
            MissingOperatorDataError, r"available values=\[\]"
        ):
            self.lookup.lookup("gemm", _key("gemm", m=2, n=7, k=9))

    def test_duplicate_and_invalid_latency_are_rejected_at_load(self):
        path = Path(self.temp.name) / "gemm_perf.parquet"
        rows = pq.read_table(path).to_pylist()
        pq.write_table(pa.Table.from_pylist(rows + [dict(rows[0])]), path)
        with self.assertRaisesRegex(OperatorDataValidationError, "duplicate exact key"):
            OperatorDataPackage.load(self.temp.name)

    def test_negative_latency_and_unknown_source_are_rejected_at_load(self):
        path = Path(self.temp.name) / "gemm_perf.parquet"
        rows = pq.read_table(path).to_pylist()
        rows[0]["latency_us"] = -1.0
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(
            OperatorDataValidationError, "finite and non-negative"
        ):
            OperatorDataPackage.load(self.temp.name)

        _generate_package(self.temp.name)
        rows = pq.read_table(path).to_pylist()
        rows[0]["source_id"] = "unknown-source"
        pq.write_table(pa.Table.from_pylist(rows), path)
        with self.assertRaisesRegex(OperatorDataValidationError, "unknown source_id"):
            OperatorDataPackage.load(self.temp.name)

    def test_old_package_schema_reports_a_migration_error(self):
        path = Path(self.temp.name) / "generation_meta.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["schema_version"] = 1
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with self.assertRaisesRegex(
            OperatorDataValidationError, "schema migration required.*expected 2"
        ):
            OperatorDataPackage.load(self.temp.name)

    def test_v2_package_row_missing_dtype_role_reports_migration_error(self):
        path = Path(self.temp.name) / "gemm_perf.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row.pop("weight_dtype")
        pq.write_table(pa.Table.from_pylist(rows), path)

        with self.assertRaisesRegex(
            OperatorDataValidationError,
            "schema v2 migration required.*weight_dtype.*regenerate",
        ):
            OperatorDataPackage.load(self.temp.name)


class OperatorDataProviderTest(unittest.TestCase):
    def setUp(self):
        self.temp = _tempdir()
        _, manifest_path = _generate_package(self.temp.name)
        package = OperatorDataPackage.load(self.temp.name)
        self.lookup = OperatorDataLookup(package)
        self.compute = OperatorDataComputeLatencyProvider(
            self.lookup, ModelOperatorManifest.load(manifest_path), "analytical"
        )
        self.transfer = OperatorDataTransferLatencyProvider(
            self.lookup, "analytical", "af-fabric", 2
        )
        self.moe_config = MoEModelConfig(
            is_moe_model=True,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            num_shared_experts=1,
            num_moe_layers=1,
            intermediate_size=16,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_old_model_manifest_reports_a_migration_error(self):
        manifest_path = Path(self.temp.name) / "legacy_model_operators.yaml"
        _write_manifest(manifest_path)
        document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        document["schema_version"] = 1
        manifest_path.write_text(yaml.safe_dump(document), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "schema migration required.*expected 2"
        ):
            ModelOperatorManifest.load(manifest_path)

    def _compute_query(
        self,
        stage="a",
        phase="decode",
        tokens=2,
        context=None,
        tp_size=1,
        ep_size=1,
        expert_workload=None,
    ):
        hidden_size = 8
        if expert_workload is None and stage == "expert":
            expert_workload = ((0, tokens),)
        return AFComputeQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase=phase,
            stage=stage,
            token_count=tokens,
            context_lengths=context if context is not None else (2,) * tokens,
            model_id="TestModel",
            dtype="bf16",
            device_type="a-device" if stage in {"a", "router"} else "f-device",
            worker_id=0 if stage in {"a", "router"} else 1,
            operation={
                "a": "a_rank_compute",
                "router": "router_topk",
                "expert": "expert_rank_compute",
            }[stage],
            source_domain=("a-domain" if stage in {"a", "router"} else "f-domain"),
            logical_participant="a0" if stage in {"a", "router"} else "f0",
            physical_worker=("a-pool:0" if stage in {"a", "router"} else "f-pool:0"),
            local_shape={
                "tokens": tokens,
                "hidden_size": (
                    hidden_size
                    if stage in {"router", "expert"}
                    else math.ceil(hidden_size / tp_size)
                ),
                "global_hidden_size": hidden_size,
                "tp_size": tp_size,
                **(
                    {"intermediate_size": 16 // tp_size}
                    if stage == "expert"
                    else {}
                ),
            },
            local_token_count=tokens,
            expert_ids=tuple(expert_id for expert_id, _ in (expert_workload or ())),
            expert_workload=tuple(expert_workload or ()),
            tp_group=tuple(range(tp_size)),
            ep_group=tuple(range(ep_size)) if stage == "expert" else (),
        )

    def test_attention_and_moe_composition_reconstruct_seconds_without_duplicates(self):
        a = self.compute.estimate(self._compute_query("a"))
        router = self.compute.estimate(self._compute_query("router"))
        expert = self.compute.estimate(self._compute_query("expert"))
        for estimate in (a, router, expert):
            total = sum(item["latency_us"] for item in estimate.details["components"])
            self.assertAlmostEqual(estimate.service_duration, total / 1e6)
            self.assertEqual(estimate.dataset_version, "test-v1")
        a_names = [item["name"] for item in a.details["components"]]
        router_names = [item["name"] for item in router.details["components"]]
        expert_names = [item["name"] for item in expert.details["components"]]
        self.assertEqual(a_names.count("qkv_projection"), 1)
        self.assertEqual(a_names.count("output_projection"), 1)
        self.assertEqual(router_names.count("router"), 1)
        self.assertEqual(expert_names.count("moe_expert"), 1)
        self.assertFalse(set(router_names) & set(expert_names))
        self.assertNotIn("dispatch", router_names + expert_names)

    def test_prefill_uses_context_attention_and_prefill_moe_rows(self):
        a = self.compute.estimate(
            self._compute_query("a", phase="prefill", tokens=2, context=(2,))
        )
        expert = self.compute.estimate(
            self._compute_query("expert", phase="prefill", tokens=2, context=(2,))
        )
        attention = next(
            item for item in a.details["components"] if item["name"] == "attention_core"
        )
        moe = next(
            item
            for item in expert.details["components"]
            if item["name"] == "moe_expert"
        )
        self.assertIn("input_seq_len", attention["key"])
        self.assertEqual(moe["key"]["phase"], "prefill")

    def test_rank_local_tp_shape_and_expert_workload_use_distinct_lookup_keys(self):
        attention = self.compute.estimate(self._compute_query("a", tp_size=2))
        by_name = {
            item["name"]: item["key"] for item in attention.details["components"]
        }
        self.assertEqual(by_name["qkv_projection"]["n"], 12)
        self.assertEqual(by_name["attention_core"]["num_heads"], 1)
        self.assertEqual(by_name["output_projection"]["k"], 4)
        self.assertEqual(attention.details["local_shape"]["hidden_size"], 4)

        expert = self.compute.estimate(
            self._compute_query("expert", tp_size=2, ep_size=2)
        )
        moe = next(
            item
            for item in expert.details["components"]
            if item["name"] == "moe_expert"
        )
        self.assertEqual(moe["key"]["num_tokens"], 2)
        self.assertEqual(moe["key"]["top_k"], 1)
        self.assertEqual(moe["key"]["num_experts"], 1)
        self.assertEqual(moe["key"]["tp_size"], 2)
        self.assertEqual(moe["key"]["ep_size"], 2)
        self.assertEqual(moe["match_type"], "interpolated")

    def test_expert_component_preserves_and_aggregates_per_expert_workload(self):
        estimate = self.compute.estimate(
            self._compute_query(
                "expert",
                tokens=3,
                expert_workload=((0, 1), (2, 2)),
            )
        )
        component = next(
            item
            for item in estimate.details["components"]
            if item["name"] == "moe_expert"
        )

        self.assertEqual(
            [
                (item["expert_id"], item["num_tokens"])
                for item in component["per_expert"]
            ],
            [(0, 1), (2, 2)],
        )
        self.assertEqual(component["latency_us"], 8.0)
        self.assertEqual(component["key"]["expert_token_counts"], "1,2")
        self.assertEqual(component["execution_unit"], "expert_group")

    def test_expert_workload_rejects_invalid_expert_metadata(self):
        out_of_range = replace(
            self._compute_query("expert"),
            expert_ids=(4,),
            expert_workload=((4, 2),),
        )
        with self.assertRaisesRegex(ValueError, "outside model range"):
            self.compute.estimate(out_of_range)

        inconsistent_ids = replace(
            self._compute_query(
                "expert",
                expert_workload=((0, 1), (2, 1)),
            ),
            expert_ids=(0, 1),
        )
        with self.assertRaisesRegex(ValueError, "does not match expert_workload"):
            self.compute.estimate(inconsistent_ids)

    def test_explicit_operator_data_config_resolves_relative_paths(self):
        config = AFConfig(
            enabled=True,
            latency_provider="operator_data",
            operator_data_path=".",
            operator_manifest_path="model_operators.yaml",
            operator_data_backend="analytical",
        )
        compute, transfer = resolve_af_latency_providers(
            config, Path(self.temp.name) / "cluster.json"
        )
        self.assertIsInstance(compute, OperatorDataComputeLatencyProvider)
        self.assertIsInstance(transfer, OperatorDataTransferLatencyProvider)

    def test_composition_conflict_and_missing_component_fail_whole_query(self):
        conflict_path = Path(self.temp.name) / "conflict.yaml"
        _write_manifest(conflict_path, duplicate_includes=True)
        provider = OperatorDataComputeLatencyProvider(
            self.lookup, ModelOperatorManifest.load(conflict_path), "analytical"
        )
        with self.assertRaisesRegex(ValueError, "covered by both"):
            provider.estimate(self._compute_query("a"))
        with self.assertRaises(MissingOperatorDataError):
            provider.estimate(self._compute_query("a", tokens=99, context=(99,)))

        missing_local_work = self._compute_query("expert")
        missing_local_work = missing_local_work.__class__(
            **{
                **missing_local_work.__dict__,
                "local_shape": {},
                "local_token_count": None,
            }
        )
        with self.assertRaisesRegex(
            ValueError, "local_token_count.*expert_rank_compute"
        ):
            provider.estimate(missing_local_work)

        split_conflict_path = Path(self.temp.name) / "split-conflict.yaml"
        _write_manifest(split_conflict_path)
        split_document = yaml.safe_load(split_conflict_path.read_text(encoding="utf-8"))
        split_document["models"]["TestModel"]["component_includes"] = {
            "router": ["moe_body"],
            "moe_expert": ["moe_body"],
        }
        split_conflict_path.write_text(yaml.safe_dump(split_document), encoding="utf-8")
        split_provider = OperatorDataComputeLatencyProvider(
            self.lookup,
            ModelOperatorManifest.load(split_conflict_path),
            "analytical",
        )
        with self.assertRaisesRegex(ValueError, "covered by both"):
            split_provider.estimate(self._compute_query("router"))

    def test_transfer_formula_payload_direction_and_service_only(self):
        query = AFTransferQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            direction="a_to_f",
            token_count=2,
            hidden_size=8,
            dtype="bf16",
            source_device="a-device",
            target_device="f-device",
            network="af-net",
            operation="af_a_to_f",
            group_size=2,
            payload_bytes=32,
        )
        first = self.transfer.estimate(query)
        second = self.transfer.estimate(query)
        self.assertEqual(first.payload_bytes, 32)
        self.assertAlmostEqual(first.service_duration, 3e-6)
        self.assertEqual(first.service_duration, second.service_duration)
        self.assertNotIn("queue_duration", first.details)
        reverse = query.__class__(
            **{
                **query.__dict__,
                "direction": "f_to_a",
                "operation": "af_f_to_a",
                "source_device": "f-device",
                "target_device": "a-device",
            }
        )
        self.assertAlmostEqual(self.transfer.estimate(reverse).service_duration, 3e-6)
        all_reduce = query.__class__(
            **{
                **query.__dict__,
                "operation": "all_reduce",
                "direction": "tp_collective",
                "target_device": "a-device",
                "resource_lane": "collective",
                "group_size": 2,
                "payload_bytes": 48,
                "participant_ranks": (0, 1),
                "logical_input_bytes_per_rank": ((0, 48), (1, 48)),
                "logical_output_bytes_per_rank": ((0, 48), (1, 48)),
            }
        )
        all_reduce_estimate = self.transfer.estimate(all_reduce)
        self.assertEqual(all_reduce_estimate.payload_bytes, 48)
        self.assertEqual(
            all_reduce_estimate.details["key"]["operation"], "all_reduce"
        )
        self.assertEqual(all_reduce_estimate.details["wire_volume_factor"], 1.0)
        self.assertEqual(all_reduce_estimate.details["planner_payload_bytes"], 48)

        all_gather = replace(
            all_reduce,
            operation="all_gather",
            logical_input_bytes_per_rank=((0, 24), (1, 24)),
            logical_output_bytes_per_rank=((0, 48), (1, 48)),
        )
        reduce_scatter = replace(
            all_reduce,
            operation="reduce_scatter",
            logical_input_bytes_per_rank=((0, 48), (1, 48)),
            logical_output_bytes_per_rank=((0, 24), (1, 24)),
        )
        self.assertEqual(self.transfer.estimate(all_gather).payload_bytes, 24)
        self.assertEqual(self.transfer.estimate(reduce_scatter).payload_bytes, 24)

        with self.assertRaisesRegex(ValueError, "unsupported.*ep_dispatch"):
            self.transfer.estimate(
                replace(query, operation="ep_dispatch", direction="ep_dispatch")
            )

    def test_transfer_key_preserves_direction_dtype_lane_and_endpoints(self):
        class RecordingLookup:
            package = SimpleNamespace(dataset_version="shape-test")

            def __init__(self):
                self.key = None

            def link_profile(self, key):
                self.key = dict(key)
                return LookupResult(
                    1.0,
                    "exact",
                    "shape-test",
                    dict(key),
                    values={
                        "fixed_latency_us": 1.0,
                        "effective_bandwidth_bytes_per_s": 1e9,
                    },
                )

        lookup = RecordingLookup()
        provider = OperatorDataTransferLatencyProvider(
            lookup, "analytical", "af-fabric", 2
        )
        query = AFTransferQuery(
            iteration_id=0,
            block_id=0,
            layer_id=0,
            phase="decode",
            direction="a_to_f",
            token_count=1,
            hidden_size=8,
            dtype="bf16",
            a_to_f_dtype="fp8",
            source_device="a-device",
            target_device="f-device",
            network="af-net",
            operation="af_a_to_f",
            resource_lane="dispatch",
            group_size=2,
            payload_bytes=8,
        )

        provider.estimate(query)
        self.assertEqual(lookup.key["transfer_dtype"], "fp8")
        self.assertEqual(lookup.key["resource_lane"], "dispatch")
        self.assertEqual(lookup.key["source_endpoint"], "a")
        self.assertEqual(lookup.key["target_endpoint"], "f")

    def test_executor_trace_preserves_provider_metadata_and_contention(self):
        cluster = ClusterConfig(
            num_workers=2,
            networks={"af-net": "test"},
            worker_groups=[
                {
                    "name": "a-pool",
                    "capabilities": ["af_attention"],
                    "hardware": "a-device",
                    "num_workers": 1,
                    "network": "af-net",
                },
                {
                    "name": "f-pool",
                    "capabilities": ["af_ffn"],
                    "hardware": "f-device",
                    "num_workers": 1,
                    "network": "af-net",
                },
            ],
            af={
                "enabled": True,
                "phases": ["prefill", "decode"],
                "af_block_size": 1,
                "links": [
                    {
                        "name": "af-link",
                        "phases": ["prefill", "decode"],
                        "attention_group": "a-pool",
                        "ffn_group": "f-pool",
                        "network": "af-net",
                        "resource_group": "af-fabric",
                        "duplex": "full",
                        "a_to_f_capacity": 1,
                        "f_to_a_capacity": 1,
                    }
                ],
            },
        )
        env = simpy.Environment()
        executor = AFExecutor(
            env,
            cluster,
            cluster.workers(),
            self.compute,
            self.transfer,
            "TestModel",
            "bf16",
            total_layers=1,
            hidden_size=8,
            moe_config=self.moe_config,
        )
        requests = [
            SimpleNamespace(
                id=index,
                is_prefill=False,
                prefill_len=2,
                prefill_compute_len=2,
                context_len=2,
                generation_idx=0,
            )
            for index in range(2)
        ]
        iteration = executor.submit(0, requests, "decode")
        env.process(executor.run_iteration(iteration))
        env.run()
        events = executor.trace.events
        self.assertEqual(len(events), 10)
        self.assertEqual(
            {event.operation for event in events if event.event_type == "transfer"},
            {"af_a_to_f", "af_f_to_a"},
        )
        self.assertTrue(all(event.dataset_version == "test-v1" for event in events))
        self.assertTrue(all(event.match_type == "exact" for event in events))
        self.assertTrue(any(event.queue_duration > 0 for event in events))
        for block_id in (0, 1):
            compute_events = [
                event
                for event in events
                if event.block_id == block_id and event.event_type == "compute"
            ]
            self.assertEqual(
                [event.stage for event in compute_events],
                ["a", "router", "expert"],
            )
            router_components = {
                item["name"] for item in compute_events[1].details["components"]
            }
            expert_components = {
                item["name"] for item in compute_events[2].details["components"]
            }
            self.assertFalse(router_components & expert_components)
            self.assertEqual(
                compute_events[1].source_domain.startswith("implicit:a-"), True
            )
            self.assertEqual(
                compute_events[2].source_domain.startswith("implicit:f-"), True
            )
            self.assertNotEqual(
                compute_events[1].physical_endpoints,
                compute_events[2].physical_endpoints,
            )
        for event in events:
            self.assertAlmostEqual(
                event.end_at - event.ready_at,
                event.queue_duration + event.service_duration,
            )

    def test_trace_toggle_does_not_change_formal_provider_timing(self):
        def run(trace_enabled):
            cluster = ClusterConfig(
                num_workers=2,
                networks={"af-net": "test"},
                worker_groups=[
                    {
                        "name": "a-pool",
                        "capabilities": ["af_attention"],
                        "hardware": "a-device",
                        "num_workers": 1,
                        "network": "af-net",
                    },
                    {
                        "name": "f-pool",
                        "capabilities": ["af_ffn"],
                        "hardware": "f-device",
                        "num_workers": 1,
                        "network": "af-net",
                    },
                ],
                af={
                    "enabled": True,
                    "phases": ["prefill", "decode"],
                    "af_block_size": 1,
                    "trace_enabled": trace_enabled,
                    "links": [
                        {
                            "name": "af-link",
                            "phases": ["prefill", "decode"],
                            "attention_group": "a-pool",
                            "ffn_group": "f-pool",
                            "network": "af-net",
                            "resource_group": "af-fabric",
                            "duplex": "full",
                            "a_to_f_capacity": 1,
                            "f_to_a_capacity": 1,
                        }
                    ],
                },
            )
            env = simpy.Environment()
            moe_config, runtime_plan = _formal_decode_runtime(cluster)
            executor = _formal_decode_executor(
                env,
                cluster,
                self.compute,
                self.transfer,
                moe_config,
                runtime_plan,
            )
            requests = [
                SimpleNamespace(
                    id=index,
                    is_prefill=False,
                    prefill_len=2,
                    prefill_compute_len=2,
                    context_len=2,
                    generation_idx=0,
                )
                for index in range(2)
            ]
            iteration = executor.submit(0, requests, "decode")
            env.process(executor.run_iteration(iteration))
            env.run()

            events = tuple(executor.trace.events_for(iteration.iteration_id))
            compute_events = [
                event for event in events if event.event_type == "compute"
            ]
            transfer_events = [
                event for event in events if event.event_type == "transfer"
            ]
            if not trace_enabled:
                return iteration, tuple(executor.trace.events)
            formal_stages = {event.stage for event in compute_events}
            self.assertTrue(
                {
                    "attention",
                    "router",
                    "routed_expert",
                    "shared_expert",
                    "weighted_combine",
                    "residual",
                }
                <= formal_stages,
                formal_stages,
            )
            self.assertFalse({"a", "expert"} & formal_stages)
            self.assertEqual(
                {event.operation for event in transfer_events},
                {"af_a_to_f", "af_f_to_a"},
            )
            self.assertTrue(any(event.match_type == "exact" for event in events))
            self.assertEqual(
                iteration.makespan,
                max(iteration.block_completed_at.values()) - iteration.start_at,
            )
            return iteration, tuple(executor.trace.events)

        enabled, events = run(True)
        disabled, no_events = run(False)
        self.assertEqual(enabled.block_completed_at, disabled.block_completed_at)
        self.assertEqual(enabled.makespan, disabled.makespan)
        self.assertTrue(events)
        self.assertFalse(no_events)

    def test_generated_package_drives_formal_decode_with_auditable_provenance(self):
        with _tempdir() as package_root:
            package, manifest_path = _generate_formal_decode_package(package_root)
            lookup = OperatorDataLookup(package)
            compute = OperatorDataComputeLatencyProvider(
                lookup,
                ModelOperatorManifest.load(manifest_path),
                "analytical",
            )
            transfer = OperatorDataTransferLatencyProvider(
                lookup, "analytical", "af-fabric", 2
            )
            cluster = _cluster(a_tp=1, a_dp=1, f_tp=1, f_dp=1, f_ep=1)
            moe_config, runtime_plan = _formal_decode_runtime(cluster)
            env = simpy.Environment()
            executor = _formal_decode_executor(
                env,
                cluster,
                compute,
                transfer,
                moe_config,
                runtime_plan,
            )
            requests = [
                Request(
                    id=request_id,
                    prefill_len=1,
                    decode_len=2,
                    block_size=2,
                )
                for request_id in (11, 12)
            ]
            for request in requests:
                request.generation_idx = 1
            iteration = executor.submit(0, requests, "decode")
            env.process(executor.run_iteration(iteration))
            env.run()

            events = tuple(executor.trace.events_for(iteration.iteration_id))
            compute_events = tuple(
                event for event in events if event.event_type == "compute"
            )
            transfer_events = tuple(
                event for event in events if event.event_type == "transfer"
            )
            self.assertTrue(events)
            self.assertEqual(
                {event.dataset_version for event in events},
                {"formal-decode-v2"},
            )
            self.assertTrue(
                all(event.match_type in {"exact", "interpolated"} for event in events)
            )
            self.assertTrue(any(event.match_type == "exact" for event in events))
            self.assertTrue(
                any(event.match_type == "interpolated" for event in compute_events)
            )
            self.assertTrue(
                {
                    "attention",
                    "router",
                    "routed_expert",
                    "shared_expert",
                    "weighted_combine",
                    "residual",
                }
                <= {event.stage for event in compute_events}
            )
            self.assertEqual(
                {event.operation for event in transfer_events},
                {"af_a_to_f", "af_f_to_a"},
            )
            self.assertFalse(
                {"ep_dispatch", "ep_combine"}
                & {event.operation for event in events}
            )
            self.assertTrue(
                all(
                    event.leg_id
                    and event.logical_stage_id
                    and event.source_endpoint is not None
                    and event.target_endpoint is not None
                    and event.payload_bytes
                    == sum(
                        int(item["payload_bytes"])
                        for item in event.payload_breakdown
                    )
                    for event in transfer_events
                )
            )
            self.assertTrue(
                any(event.queue_duration > 0 for event in transfer_events)
            )
            for event in compute_events:
                components = tuple(event.details.get("components", ()))
                self.assertTrue(components)
                self.assertTrue(
                    all(
                        component["source_id"]
                        and component["match_type"] in {"exact", "interpolated"}
                        for component in components
                    )
                )
                self.assertAlmostEqual(
                    event.service_duration,
                    sum(float(item["latency_us"]) for item in components) / 1e6,
                )
            self.assertEqual(
                iteration.makespan,
                max(iteration.block_completed_at.values()) - iteration.start_at,
            )

    def test_generated_package_runs_prefill_and_decode_through_llm_engine(self):
        reset_g_time()
        cluster = ClusterConfig(
            num_workers=2,
            networks={"af-net": "profile"},
            worker_groups=[
                {
                    "name": "a-pool",
                    "capabilities": ["af_attention"],
                    "hardware": "a-device",
                    "num_workers": 1,
                    "network": "af-net",
                },
                {
                    "name": "f-pool",
                    "capabilities": ["af_ffn"],
                    "hardware": "f-device",
                    "num_workers": 1,
                    "network": "af-net",
                },
            ],
            af={
                "enabled": True,
                "phases": ["prefill", "decode"],
                "links": [
                    {
                        "name": "af-link",
                        "phases": ["prefill", "decode"],
                        "attention_group": "a-pool",
                        "ffn_group": "f-pool",
                        "network": "af-net",
                        "resource_group": "af-fabric",
                        "duplex": "full",
                        "a_to_f_capacity": 1,
                        "f_to_a_capacity": 1,
                    }
                ],
            },
        )
        env = simpy.Environment()
        roofline = AFRooflineStub()
        roofline.models["TestModel"].Dmodel = 8
        roofline.models["TestModel"].Nhead = 2
        with patch("TokenSim.llm.llm_engine.CacheConfig", LightweightCacheConfig):
            engine = LLMEngine(
                env=env,
                block_size=2,
                batching="dynamic",
                kv_transfer_config=KVTransferConfig.default(),
                psla_config=replace(
                    _psla_config(),
                    is_moe_model=True,
                    num_experts=4,
                    num_experts_per_tok=2,
                    num_moe_layers=1,
                    moe_intermediate_size=16,
                    num_shared_experts=1,
                    intermediate_size=16,
                    hidden_size=8,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                ),
                cluster_config=cluster,
                roofline=roofline,
                prefill_worker_pool_type="round_robin",
                decode_worker_pool_type="round_robin",
                max_parallem_sum=8,
                af_compute_provider=self.compute,
                af_transfer_provider=self.transfer,
                af_dtype="bf16",
            )
        request = Request(id=100, prefill_len=2, decode_len=2, block_size=2)
        env.process(
            LLMSource(
                env=env,
                engine=engine,
                requests=[request],
                qps=1,
                distribution="burst",
            )
        )

        def stop_when_done():
            while env.now < 0.1:
                if request.is_done:
                    engine.send_task(engine, Task.STOP)
                    yield env.timeout(1e-9)
                    return
                yield env.timeout(1e-6)
            raise AssertionError("operator-data AF request did not finish")

        monitor = env.process(stop_when_done())
        env.run(until=monitor)
        self.assertTrue(request.is_done)
        self.assertEqual(
            [summary["phase"] for summary in engine.af_iteration_summaries],
            ["prefill", "decode"],
        )
        self.assertTrue(
            all(
                event["dataset_version"] == "test-v1"
                for summary in engine.af_iteration_summaries
                for event in summary["events"]
            )
        )


if __name__ == "__main__":
    unittest.main()
