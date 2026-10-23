"""KV-parallel domain configuration and runtime compiler validation."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from TokenSim.af.factory import build_model_runtime
from TokenSim.config.af.domains import DomainParallelConfig, ExecutionDomainConfig
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration


class KVParallelConfigTest(unittest.TestCase):
    def test_defaults_preserve_existing_domains(self):
        parallel = DomainParallelConfig(tensor_parallel_size=4)
        self.assertEqual(
            (
                parallel.kv_parallel_size,
                parallel.kv_parallel_layers,
                parallel.kv_append_block,
            ),
            (1, "full_attention", 16),
        )
        self.assertEqual(parallel.ranks_per_dp_group, 4)
        explicit = DomainParallelConfig(
            tensor_parallel_size=16, kv_parallel_size=4, kv_parallel_layers="all"
        )
        self.assertEqual(explicit.world_size, 16)

    def test_invalid_fields_raise_configuration_errors(self):
        for fields in (
            dict(tensor_parallel_size=4, kv_parallel_size=3),
            dict(tensor_parallel_size=4, kv_parallel_size=0),
            dict(tensor_parallel_size=4, kv_append_block=0),
            dict(tensor_parallel_size=4, kv_parallel_layers="windowed"),
        ):
            with self.subTest(fields=fields), self.assertRaises(ConfigurationError):
                DomainParallelConfig(**fields)

    def test_only_attention_domains_declare_kv_parallelism(self):
        parallel = DomainParallelConfig(tensor_parallel_size=2, kv_parallel_size=2)
        ExecutionDomainConfig(
            name="a", capability="af_attention", worker_group="g", parallel=parallel
        )
        with self.assertRaises(ConfigurationError):
            ExecutionDomainConfig(
                name="f", capability="af_ffn", worker_group="g", parallel=parallel
            )


class KVParallelCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[3] / "data"
        )
        cls.model = replace(
            cls.hardware.model("gemma-4-31b-text"),
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            attention_windows=(),
            attention_geometries=(),
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
        )

    def runtime(self, kv_parallel_size, tp=2):
        def build(doc, **kwargs):
            doc["af"]["execution_domains"][0]["parallel"][
                "kv_parallel_size"
            ] = kv_parallel_size
            return build_model_runtime(doc, **kwargs)

        layout = Layout(
            a_device="groq_lp30", f_device="groq_lp30", a_tp=tp, f_tp=tp, f_ep=1
        )
        with patch("scripts.lpx.native_config.build_model_runtime", build):
            return configuration(self.hardware, self.model, layout, namespace="kvp")

    def test_compiler_records_kv_parallelism_and_rejects_uneven_heads(self):
        domain = self.runtime(2).plan.domain_runtime("kvp-a-domain")
        self.assertEqual(
            (domain.kv_parallel_size, domain.kv_parallel_layers), (2, "full_attention")
        )
        self.assertEqual(
            self.runtime(1).plan.domain_runtime("kvp-a-domain").kv_parallel_size, 1
        )
        self.model = replace(self.model, num_attention_heads=2, num_key_value_heads=1)
        with self.assertRaisesRegex(ConfigurationError, "divisible"):
            self.runtime(2, tp=4)


if __name__ == "__main__":
    unittest.main()
