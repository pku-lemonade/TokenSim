"""Per-layer attention and packed expert storage retain precise ownership."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.config.model_config import ModelSpec
from TokenSim.dtypes import canonicalize_dtype, dtype_size_bytes
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.weight_storage import weight_storage_bytes
from TokenSim.operator_data.analytical.base import analytical_model_for
from TokenSim.operator_data.workload import work_for_table
from tests.af.support.config import _cluster
from tests.af.support.model import (
    _compute_query,
    _runtime_hardware,
    _runtime_plan,
    _moe,
)


class PrecisionLayoutTest(unittest.TestCase):
    def test_packed_formats_are_weight_only_and_store_row_scale_metadata(self):
        self.assertEqual(dtype_size_bytes("mxfp4", role="weight_dtype"), 0.5)
        for role in (
            "activation_dtype",
            "kv_cache_dtype",
            "a_to_f_dtype",
            "f_to_a_dtype",
        ):
            with self.assertRaises(ValueError):
                canonicalize_dtype("mxfp4", role=role)
        self.assertEqual(weight_storage_bytes((2, 33), "mxfp4"), (34, 4))
        self.assertEqual(weight_storage_bytes((2, 32), "mxfp4"), (32, 2))
        with self.assertRaises(ConfigurationError):
            weight_storage_bytes((2, 0), "mxfp4")

    def test_runtime_accounts_for_expert_scales_and_preserves_attention_dtype(self):
        hardware = _runtime_hardware()
        model = replace(
            hardware.model("TestModel"),
            moe=_moe(top_k=1),
            operator_dtypes={"routed_experts": "mxfp4"},
        )
        from TokenSim.af.runtime.compiler import compile_af_runtime_plan

        cluster = _cluster(a_tp=1, a_dp=1, f_tp=2, f_dp=2, f_ep=2)
        plan = compile_af_runtime_plan(
            cluster_config=cluster,
            worker_configs=cluster.workers(),
            hardware=hardware,
            model=model,
            block_size=4,
        )
        experts = [
            c
            for s in plan.physical_states
            for c in s.components
            if c.category == "routed_expert"
        ]
        self.assertTrue(
            all(c.storage_dtype == "mxfp4" and c.scale_bytes == 48 for c in experts)
        )
        self.assertEqual(sum(c.bytes for c in experts), 3456)
        attention = [
            c
            for s in plan.physical_states
            for c in s.components
            if c.category == "attention"
        ]
        self.assertTrue(all(c.storage_dtype is None for c in attention))

    def test_layer_windows_drive_native_attention_queries(self):
        hardware = _runtime_hardware()
        model = replace(hardware.model("TestModel"), attention_windows=(128, 0))
        provider = OperatorTableAFComputeProvider(
            hardware, model, fallback="analytical_only"
        )
        query = _compute_query("a", tokens=1, context=(1024,))
        local = provider.estimate(replace(query, layer_id=0))
        full = provider.estimate(replace(query, layer_id=1))
        local_core = next(
            c for c in local.details["components"] if c["name"] == "attention_core"
        )
        full_core = next(
            c for c in full.details["components"] if c["name"] == "attention_core"
        )
        self.assertEqual(local_core["key"]["window_size"], 128)
        self.assertEqual(full_core["key"]["window_size"], 0)
        self.assertLess(local.service_duration, full.service_duration)
        self.assertEqual(model.layer_specs()[0].attention_kind, "sliding")
        with self.assertRaises(ConfigurationError):
            replace(model, attention_windows=(128,))
        with self.assertRaises(ConfigurationError):
            replace(model, attention_windows=(True, 0))

    def test_lp30_packed_execution_uses_fp8_fallback_with_dequant_work(self):
        hardware = HardwareContext.load(Path(__file__).resolve().parents[3] / "data")
        device = hardware.device("groq_lp30")
        self.assertEqual(
            device.peak_compute_for("mxfp4").value, device.peak_compute_for("fp8").value
        )
        key = dict(
            dtype="mxfp4",
            activation_dtype="bf16",
            hidden_size=16,
            inter_size=32,
            expert_tokens="[1,2]",
        )
        work = work_for_table("routed_experts", key)
        self.assertGreater(work.details["weight_scale_bytes"], 0)
        estimate = analytical_model_for("groq_tsp").estimate(
            "routed_experts", key, device
        )
        self.assertGreater(estimate.details["packed_dequant_time_us"], 0)
        self.assertEqual(estimate.work.weight_bytes, work.weight_bytes)

    def test_family_registration_is_independent_of_import_order(self):
        from TokenSim.operator_data.analytical import base, gpu
        from types import SimpleNamespace

        with patch.dict(
            base._MODELS, {"nvidia_gpu": gpu.GPURooflineModel()}, clear=True
        ):
            self.assertEqual(base.analytical_model_for("groq_tsp").family, "groq_tsp")
        override = SimpleNamespace(family="groq_tsp")
        with patch.dict(base._MODELS, {"groq_tsp": override}, clear=True):
            self.assertIs(base.analytical_model_for("groq_tsp"), override)

    def test_verification_attention_charges_all_query_positions(self):
        hardware = _runtime_hardware()
        model = hardware.model("TestModel")
        provider = OperatorTableAFComputeProvider(
            hardware, model, fallback="analytical_only"
        )
        query = replace(
            _compute_query("a", tokens=5, context=(1029,)), query_lengths=(5,)
        )
        estimate = provider.estimate(query)
        core = next(
            component
            for component in estimate.details["components"]
            if component["name"] == "attention_core"
        )
        self.assertEqual(core["table"], "context_attention")
        self.assertEqual(core["key"]["input_seq_len"], 5)
        self.assertEqual(core["key"]["context_len"], 1029)
