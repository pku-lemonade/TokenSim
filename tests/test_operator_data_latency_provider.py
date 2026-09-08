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
from TokenSim.af import (
    AFComputeQuery,
    AFExecutor,
    AFTransferQuery,
    resolve_af_latency_providers,
)
from TokenSim.config.cluster_config import AFConfig
from TokenSim.config.config import KVTransferConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.llm.llm_engine import LLMEngine, Task
from TokenSim.llm.llm_request import Request, reset_g_time
from TokenSim.moe.config import MoEModelConfig
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
        peak_compute_by_dtype={"bf16": 1e12},
        memory_bandwidth_bytes_per_s=1e11,
        memory_capacity_bytes=230_000_000,
        interconnect_bandwidth_bytes_per_s=480_000_000_000,
    )


def _key(table, device="a-device", **overrides):
    values = {
        "gemm": {
            "device": device,
            "backend": "analytical",
            "gemm_dtype": "bf16",
            "m": 2,
            "n": 24,
            "k": 8,
        },
        "context_attention": {
            "device": device,
            "backend": "analytical",
            "attn_dtype": "bf16",
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
            "attn_dtype": "bf16",
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
            "moe_dtype": "bf16",
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
            "dtype": "bf16",
            "num_tokens": 2,
            "hidden_size": 8,
        },
        "communication": {
            "source_device": "a-device",
            "target_device": "f-device",
            "backend": "analytical",
            "network": "af-net",
            "link_group": "af-fabric",
            "operation": "af_transfer",
            "direction": "a_to_f",
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

    for operation, direction, source_device, target_device, group_size in (
        ("af_a_to_f", "a_to_f", "a-device", "f-device", 2),
        ("af_f_to_a", "f_to_a", "f-device", "a-device", 2),
        ("ep_dispatch", "ep_dispatch", "f-device", "f-device", 1),
        ("ep_combine", "ep_combine", "f-device", "f-device", 1),
    ):
        key = _key(
            "communication",
            operation=operation,
            source_device=source_device,
            target_device=target_device,
            direction=direction,
            group_size=group_size,
        )
        add("communication", key, 0.0)
    for operation, direction, source_device, target_device, group_size in (
        ("tp_collective", "tp_collective", "a-device", "a-device", 2),
        ("tp_collective", "tp_collective", "f-device", "f-device", 2),
    ):
        add(
            "communication",
            _key(
                "communication",
                operation=operation,
                direction=direction,
                source_device=source_device,
                target_device=target_device,
                group_size=group_size,
            ),
            0.0,
        )
    return source, shapes, observations


def _write_manifest(path, duplicate_includes=False):
    includes = {}
    if duplicate_includes:
        includes = {
            "qkv_projection": ["projection"],
            "output_projection": ["projection"],
        }
    document = {
        "schema_version": 1,
        "models": {
            "TestModel": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_experts": 4,
                "top_k": 2,
                "tp_size": 1,
                "ep_size": 1,
                "window_size": 0,
                "kv_cache_dtype": "bf16",
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
                    intermediate_size=16,
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
                "hidden_size": 4,
                "global_hidden_size": 8,
                "tp_size": 2,
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

    def test_unknown_source_and_schema_are_rejected_at_load(self):
        path = Path(self.temp.name) / "generation_meta.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["schema_version"] = 99
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with self.assertRaisesRegex(OperatorDataValidationError, "unsupported schema"):
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
            num_moe_layers=1,
        )

    def tearDown(self):
        self.temp.cleanup()

    def _compute_query(
        self,
        stage="a",
        phase="decode",
        tokens=2,
        context=(2,),
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
            context_lengths=context,
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
                    if stage == "router"
                    else math.ceil(hidden_size / tp_size)
                ),
                "global_hidden_size": hidden_size,
                "tp_size": tp_size,
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
        self.assertEqual(component["latency_us"], 18.0)

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
        dispatch = query.__class__(
            **{
                **query.__dict__,
                "operation": "ep_dispatch",
                "direction": "ep_dispatch",
                "source_device": "f-device",
                "target_device": "f-device",
                "group_size": 1,
                "payload_bytes": 64,
            }
        )
        dispatch_estimate = self.transfer.estimate(dispatch)
        self.assertEqual(dispatch_estimate.payload_bytes, 64)
        self.assertEqual(dispatch_estimate.details["key"]["operation"], "ep_dispatch")
        all_reduce = query.__class__(
            **{
                **query.__dict__,
                "operation": "tp_collective",
                "direction": "tp_collective",
                "target_device": "a-device",
                "group_size": 2,
                "payload_bytes": 48,
            }
        )
        all_reduce_estimate = self.transfer.estimate(all_reduce)
        self.assertEqual(all_reduce_estimate.payload_bytes, 48)
        self.assertEqual(
            all_reduce_estimate.details["key"]["operation"], "tp_collective"
        )

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
                )
                for index in range(2)
            ]
            iteration = executor.submit(0, requests, "decode")
            env.process(executor.run_iteration(iteration))
            env.run()
            return iteration, tuple(executor.trace.events)

        enabled, events = run(True)
        disabled, no_events = run(False)
        self.assertEqual(enabled.block_completed_at, disabled.block_completed_at)
        self.assertEqual(enabled.makespan, disabled.makespan)
        self.assertTrue(events)
        self.assertFalse(no_events)

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
