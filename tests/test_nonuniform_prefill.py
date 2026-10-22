"""Prefill, decode and admission share explicit contiguous layer ownership."""

from dataclasses import replace
from pathlib import Path
import unittest

import simpy

from TokenSim.af.serving.service import AFService
from TokenSim.config.attention_geometry import AttentionGeometry
from TokenSim.config.cache_config import stage_layer_range
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.llm_request import Request
from scripts.lpx.config import Layout
from scripts.lpx.native_config import configuration


class NonuniformPrefillTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = ModelSpec(
            "nonuniform-prefill",
            32,
            64,
            5,
            4,
            2,
            8,
            vocab_size=64,
            attention_windows=(4, 0, 4, 0, 4),
            attention_geometries=tuple(
                AttentionGeometry(16, 1, True) if i % 2 else AttentionGeometry(8, 2)
                for i in range(5)
            ),
            max_position_embeddings=1024,
            dtype="bf16",
        )
        cls.layout = Layout(
            a_device="groq_lp30",
            f_device="groq_lp30",
            a_tp=2,
            f_tp=2,
            a_pp=2,
            f_pp=2,
            f_ep=1,
        )

    def runtime(self, counts):
        return configuration(
            self.hardware,
            self.model,
            self.layout,
            namespace="prefill",
            micro_batch=2,
            pipeline_layer_counts=counts,
            colocated_dense_reuse=True,
        )

    def run_chunks(self, runtime):
        env = simpy.Environment()
        service = AFService(env, runtime)
        session = service.admit(0, 0, maximum_context=8, current_context=0)
        for queries, logits in ((2, False), (2, False), (1, True)):
            env.run(
                until=env.process(
                    service.forward(
                        [session],
                        phase="prefill",
                        query_tokens=queries,
                        emit_logits=logits,
                    )
                )
            )
        self.assertEqual(session.cached_context, 5)
        env.run(until=env.process(service.forward([session], phase="decode")))
        self.assertEqual(session.cached_context, 6)
        service.release(session)
        self.assertFalse(service.memory.leases)
        return env.now, service

    def test_chunked_prefill_uses_native_stage_and_geometry_ownership(self):
        _, service = self.run_chunks(self.runtime((1, 4)))
        self.assertEqual(len(service.prefill.backends), 4)
        for rank_key, backend in service.prefill.backends.items():
            binding = service.runtime.plan.binding(*rank_key)
            owned = stage_layer_range(5, 2, binding.pp_rank, (1, 4))
            self.assertEqual(backend.layers_local, len(owned))
            grouped = {}
            for child in backend._layer_backends:
                key = (
                    child.model.sliding_window,
                    child.model.head_dim,
                    child.model.num_key_value_heads,
                )
                grouped[key] = child.layers_local
            expected = {}
            for layer in owned:
                geometry = self.model.attention_geometry(layer)
                key = (
                    self.model.attention_window(layer),
                    geometry.head_dim,
                    geometry.num_key_value_heads,
                )
                expected[key] = expected.get(key, 0) + 1
            self.assertEqual(grouped, expected)
            self.assertEqual(backend.stats.step_count, 3)

    def test_explicit_balanced_partition_preserves_prefill_and_decode(self):
        implicit_time, implicit = self.run_chunks(self.runtime(None))
        explicit_time, explicit = self.run_chunks(self.runtime((3, 2)))
        self.assertEqual(implicit_time, explicit_time)
        self.assertEqual(implicit.trace.events, explicit.trace.events)

    def test_homogeneous_custom_stage_charges_its_actual_layer_count(self):
        model = replace(self.model, attention_windows=(), attention_geometries=())
        latencies = []
        for counts in ((1, 4), (4, 1)):
            backend = OperatorTableLatencyBackend(
                device=self.hardware.device("groq_lp30"),
                model=model,
                parallel_config=ParallelConfig(pipeline_parallel_size=2),
                pipeline_layer_counts=counts,
                fallback="analytical_only",
            )
            backend.include_pp_transfer = backend.include_step_overhead = False
            self.assertEqual(backend.layers_local, counts[0])
            self.assertFalse(backend._layer_backends)
            latencies.append(backend.estimate_step_latency([Request(0, 16, 1, 16)]))
        self.assertAlmostEqual(latencies[1], 4 * latencies[0], places=14)

    def test_invalid_or_unsupported_backend_partitions_fail_before_estimation(self):
        settings = dict(
            device=self.hardware.device("groq_lp30"),
            parallel_config=ParallelConfig(pipeline_parallel_size=2),
            rank_info=ParallelRankInfo(),
            fallback="analytical_only",
        )
        for counts in ((1, 3), (0, 5), (True, 4)):
            with self.subTest(counts=counts), self.assertRaises(ConfigurationError):
                OperatorTableLatencyBackend(
                    model=self.model, pipeline_layer_counts=counts, **settings
                )
        compressed = self.hardware.model("gpt-oss-2t-wide-proxy")
        with self.assertRaisesRegex(ConfigurationError, "dense ordinary"):
            OperatorTableLatencyBackend(
                model=compressed,
                pipeline_layer_counts=(1, compressed.num_layers - 1),
                **settings,
            )

    def test_default_homogeneous_backend_retains_empty_stage_support(self):
        model = replace(self.model, attention_windows=(), attention_geometries=())
        backend = OperatorTableLatencyBackend(
            device=self.hardware.device("groq_lp30"),
            model=model,
            parallel_config=ParallelConfig(pipeline_parallel_size=6),
            rank_info=ParallelRankInfo(pp_rank=5),
            fallback="analytical_only",
        )
        self.assertEqual(backend.layers_local, 0)
        self.assertTrue(backend.is_last_stage)
        self.assertFalse(backend._layer_backends)
        self.assertGreater(backend.estimate_step_latency([Request(0, 16, 1, 16)]), 0)


if __name__ == "__main__":
    unittest.main()
