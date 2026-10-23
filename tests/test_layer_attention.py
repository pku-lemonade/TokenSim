"""Mixed-layer geometry, block operations, pipeline ownership and KV accounting."""

import simpy
import unittest
from TokenSim.af.runtime.admission import rank_kv_bytes
from TokenSim.af.serving.handoff import plan_handoff
from TokenSim.af.serving.service import AFService
from TokenSim.config.attention_geometry import AttentionGeometry
from TokenSim.config.cache_config import CacheConfig, stage_layer_range
from TokenSim.config.components import TransformerBlockLayout
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.llm_request import Request
from dataclasses import replace
from pathlib import Path
from scripts.lpx_runtime import Layout, configuration
from unittest.mock import patch


class LayerAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )
        cls.model = ModelSpec(
            "mixed",
            32,
            64,
            2,
            4,
            2,
            8,
            vocab_size=64,
            attention_windows=(4, 0),
            attention_geometries=(
                AttentionGeometry(8, 2),
                AttentionGeometry(16, 1, True),
            ),
            dtype="bf16",
            max_position_embeddings=1024,
        )

    def test_projection_sharing_preserves_both_transformed_caches(self):
        runtime = configuration(
            self.hardware, self.model, Layout(a_tp=2, f_ep=2), namespace="mixed"
        )
        binding = next(
            b for b in runtime.plan.rank_bindings if b.capability == "af_attention"
        )
        self.assertIn(
            "communication_workspace",
            dict(
                runtime.plan.physical_state(binding.physical_identity).protocol_buffers
            ),
        )
        local = runtime.plan.local_shape(
            binding.domain, binding.domain_rank, 1, "attention"
        )
        self.assertEqual(
            (local["head_dim"], local["num_kv_heads"], local["kv_projections"]),
            (16, 1, 1),
        )
        # Local layer: 4 cached positions * (K+V) * 1 head * 8 dims * BF16.
        # Global layer: 100 positions * (K+V) * 1 replicated head * 16 dims * BF16.
        self.assertEqual(rank_kv_bytes(runtime.plan, binding, 100), 4 * 32 + 100 * 64)
        components = [
            c
            for c in runtime.plan.physical_state(binding.physical_identity).components
            if c.category == "attention"
        ]
        weights = {c.layer_id: c.bytes for c in components}
        self.assertEqual(
            weights, {0: (2 * 32 * 16 + 2 * 32 * 8) * 2, 1: (2 * 32 * 32 + 32 * 16) * 2}
        )

    def test_prefill_queries_use_each_window_and_geometry(self):
        backend = OperatorTableLatencyBackend(
            device=self.hardware.device("rubin"),
            model=self.model,
            fallback="analytical_only",
        )
        requests = [Request(0, 32, 2, 16)]
        observed = []
        original = OperatorTableLatencyBackend.query

        def query(owner, table, key):
            if table == "context_attention":
                observed.append(dict(key))
            return original(owner, table, key)

        with patch.object(OperatorTableLatencyBackend, "query", query):
            latency = backend.estimate_step_latency(requests)
        self.assertGreater(latency, 0)
        self.assertEqual(
            {(k["window_size"], k["head_dim"], k["num_kv_heads"]) for k in observed},
            {(4, 8, 2), (0, 16, 1)},
        )
        self.assertEqual(backend.stats.step_count, 1)

    def test_cached_prefill_uses_kv_reads_without_scaling_launch_overhead(self):
        model = replace(self.model, attention_geometries=(), attention_windows=())
        backend = OperatorTableLatencyBackend(
            device=self.hardware.device("rubin"),
            model=model,
            fallback="analytical_only",
        )
        actual = backend._attention_prefill(1, 8, 400008)
        expected = backend.query(
            "context_attention",
            dict(
                attn_dtype=model.activation_dtype,
                kv_cache_dtype=model.kv_cache_dtype,
                batch_size=1,
                input_seq_len=8,
                context_len=400008,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                window_size=0,
            ),
        ).latency_us
        self.assertEqual(actual, expected)
        self.assertLess(actual, 100)

    def test_mxfp4_experts_reach_prefill_backend(self):
        model = replace(
            self.hardware.model("moe-toy"),
            attention_windows=(4, 0, 4, 0),
            operator_dtypes={"routed_experts": "mxfp4"},
        )
        backend = OperatorTableLatencyBackend(
            device=self.hardware.device("rubin"),
            model=model,
            fallback="analytical_only",
        )
        self.assertEqual(backend.moe_dtype, "mxfp4")
        self.assertTrue(backend._layer_backends)
        self.assertTrue(
            all(child.moe_dtype == "mxfp4" for child in backend._layer_backends)
        )

    def test_catalog_geometry_and_parameter_count(self):
        model = self.hardware.model("gemma-4-31b-text")
        self.assertEqual(sum(model.attention_window(i) == 0 for i in range(60)), 10)
        self.assertEqual(model.attention_geometry(5), AttentionGeometry(512, 4, True))
        self.assertTrue(model.gated)
        self.assertTrue(30_000_000_000 < model.total_params() < 32_000_000_000)
        with self.assertRaises(ConfigurationError):
            replace(self.model, attention_geometries=(AttentionGeometry(8, 2),))

    def test_uniform_block_cache_accounts_for_each_layers_kv_geometry(self):
        for pp, stage, expected in ((1, 0, 96), (2, 0, 32), (2, 1, 64)):
            with self.subTest(pp=pp, stage=stage):
                cache = CacheConfig(
                    block_size=16,
                    model=self.model,
                    device=self.hardware.device("rubin"),
                    parallel_config=ParallelConfig(
                        tensor_parallel_size=2, pipeline_parallel_size=pp
                    ),
                    rank_info=ParallelRankInfo(pp_rank=stage),
                )
                self.assertEqual(cache.size_per_token, expected)

    def test_block_operations_depend_on_declared_structure_and_not_model_name(self):
        structure = TransformerBlockLayout(
            attention_output_norm=True,
            ffn_output_norm=True,
            value_norm=True,
            embedding_scale=True,
            layer_scale=True,
            logit_softcap=30.0,
        )
        traces = []
        for architecture in ("transformer", "arbitrary-model-family"):
            model = replace(
                self.model, architecture=architecture, block_layout=structure
            )
            runtime = configuration(
                self.hardware, model, Layout(a_tp=2, f_ep=2), namespace="structure"
            )
            extra_parameters = 2 * model.hidden_size + 1
            self.assertEqual(
                model.total_params() - self.model.total_params(),
                model.num_layers * extra_parameters,
            )
            for state in runtime.plan.physical_states:
                for component in state.components:
                    if component.category == "attention":
                        self.assertEqual(
                            dict(component.shape)["block_norm_elements"],
                            extra_parameters,
                        )
            base_cache = CacheConfig(
                block_size=16,
                model=self.model,
                device=self.hardware.device("rubin"),
                parallel_config=ParallelConfig(tensor_parallel_size=2),
            )
            extra_cache = CacheConfig(
                block_size=16,
                model=model,
                device=self.hardware.device("rubin"),
                parallel_config=ParallelConfig(tensor_parallel_size=2),
            )
            self.assertEqual(
                extra_cache.model_param_size - base_cache.model_param_size,
                model.num_layers * extra_parameters * 2,
            )
            env = simpy.Environment()
            service = AFService(env, runtime)
            session = service.admit(0, 0, maximum_context=24, current_context=16)
            env.run(until=env.process(service.forward([session])))
            traces.append(service.trace.events)
            names = {
                c["name"]
                for e in service.trace.events
                for c in e.details.get("components", [])
            }
            self.assertTrue(
                {
                    "post_attn_norm",
                    "post_ffn_norm",
                    "v_norm",
                    "embedding_scale",
                    "layer_scalar",
                    "logit_softcap",
                }.issubset(names)
            )
            for event in service.trace.events:
                ordered = [
                    component["name"]
                    for component in event.details.get("components", [])
                ]
                if "v_norm" in ordered:
                    self.assertLess(
                        ordered.index("v_norm"), ordered.index("attention_core")
                    )
        self.assertEqual(traces[0], traces[1])

    def test_mixed_layer_prefill_charges_embedding_scale_once(self):
        requests = [Request(0, 16, 1, 16)]
        base = OperatorTableLatencyBackend(
            device=self.hardware.device("rubin"),
            model=self.model,
            fallback="analytical_only",
        )
        model = replace(
            self.model, block_layout=TransformerBlockLayout(embedding_scale=True)
        )
        scaled = OperatorTableLatencyBackend(
            device=self.hardware.device("rubin"),
            model=model,
            fallback="analytical_only",
        )
        difference = scaled.estimate_step_latency(
            requests
        ) - base.estimate_step_latency(requests)
        expected = base.elementwise("scale", 16, self.model.hidden_size) / 1e6
        self.assertAlmostEqual(difference, expected, places=13)
        self.assertEqual(model.to_dict()["block_layout"], {"embedding_scale": True})
        self.assertNotIn("block_layout", self.model.to_dict())
        for invalid in ({"value_norm": 1}, {"logit_softcap": -1}, {"unknown": True}):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigurationError):
                replace(self.model, block_layout=invalid)


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
        input_cost = (
            backend.elementwise("embedding_lookup", 16, model.hidden_size) * 1e-6
        )
        self.assertAlmostEqual(
            latencies[1] - input_cost, 4 * (latencies[0] - input_cost), places=14
        )

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
        compressed = self.hardware.model("deepseek-v4-pro")
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

    def test_kv_handoff_preserves_layer_geometry_across_stage_boundaries(self):
        source = configuration(
            self.hardware,
            self.model,
            replace(self.layout, a_pp=3, f_pp=3),
            namespace="source",
            pipeline_layer_counts=(2, 2, 1),
        )
        target = self.runtime((1, 4))
        env = simpy.Environment()
        before_service = AFService(env, source, trace=False)
        after_service = AFService(env, target, trace=False)
        before = before_service.admit(1, 0, maximum_context=24, current_context=16)
        after = after_service.admit(1, 0, maximum_context=24, current_context=0)
        copies = plan_handoff(
            before_service,
            before,
            after_service,
            after,
            context=16,
            mode="full",
            cross_network="cross",
            replication_network="replication",
        )
        expected = sum(
            rank_kv_bytes(target.plan, binding, 16)
            for binding in target.plan.rank_bindings
            if binding.kv_partition
        )
        self.assertEqual(sum(copy.bytes for copy in copies), expected)
        self.assertEqual({copy.layer_id for copy in copies}, set(range(5)))
        before_service.release(before)
        after_service.release(after)
