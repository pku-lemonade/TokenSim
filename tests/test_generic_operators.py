"""Operator plans depend on geometry and data bindings, independent of model names."""

from dataclasses import replace
from pathlib import Path
import unittest

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.config.kv_layout import hybrid_kv_layout
from TokenSim.hardware import HardwareContext
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.latency.projection import attention_projections
from TokenSim.operator_data.package import OperatorDataPackage
from TokenSim.llm.forward import ForwardPass, RankShape, sp_valid_rows
from TokenSim.moe.placement import build_expert_placement
from tests.support.hardware import make_model as make_model


class GenericOperatorTest(unittest.TestCase):
    def test_cache_identity_tracks_geometry_independently_of_latency_bindings(self):
        model = make_model()
        self.assertEqual(
            model.cache_signature(),
            replace(
                model, operator_modules={"attention": {"table": "other"}}
            ).cache_signature(),
        )
        self.assertNotEqual(
            model.cache_signature(), replace(model, num_layers=5).cache_signature()
        )
        self.assertNotEqual(
            model.cache_signature(),
            replace(model, kv_cache_dtype="fp8").cache_signature(),
        )

    @classmethod
    def setUpClass(cls):
        cls.hardware = HardwareContext.load(
            Path(__file__).resolve().parents[1] / "data"
        )

    def test_projection_plan_supports_dense_and_grouped_attention(self):
        dense = make_model(num_attention_heads=8, num_key_value_heads=2)
        projections = attention_projections(dense, 4)
        self.assertEqual(
            projections[0].key(7), {"dtype": "fp16", "m": 7, "n": 64, "k": 128}
        )
        self.assertEqual(
            projections[1].key(7), {"dtype": "fp16", "m": 7, "n": 128, "k": 32}
        )
        model = replace(
            self.hardware.model("DeepSeek-V4-Pro"),
            model_id="custom-compressed-model",
            aliases=(),
        )
        plan = attention_projections(model, 4)
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan[2].groups, model.components.o_groups // 4)

    def test_fused_moe_supports_other_model_architectures(self):
        source = self.hardware.model("DeepSeek-V4-Pro")
        other = replace(
            source,
            model_id="custom-moe",
            aliases=(),
            architecture="transformer",
            components=None,
            operator_modules={
                **source.operator_modules,
                "moe": {
                    **source.operator_modules["moe"],
                    "key": {
                        **source.operator_modules["moe"]["key"],
                        "distribution": "alternate_profile",
                    },
                },
            },
        )
        parallel = ParallelConfig(data_parallel_size=8, enable_expert_parallel=True)
        package = self.hardware.operator_package(
            self.hardware.device("b200").device_id, "vllm"
        )
        step = ForwardPass({rank: RankShape(16 + rank, 1, True) for rank in range(8)})
        estimates = []
        alternate = OperatorDataPackage.from_rows(
            package.meta,
            {
                "dsv4_megamoe": [
                    {**row, "distribution": "alternate_profile"}
                    for row in package.tables["dsv4_megamoe"]
                    if row["distribution"]
                    == source.operator_modules["moe"]["key"]["distribution"]
                ]
            },
        )
        for model, data in ((source, package), (other, alternate)):
            backend = OperatorTableLatencyBackend(
                device=self.hardware.device("b200"),
                model=model,
                parallel_config=parallel,
                package=data,
                moe_execution={"backend": "deep_gemm_amxf4_mega_moe"},
            )
            backend.set_moe_step(step)
            estimates.append(backend.native_moe.estimate(16))
        self.assertEqual(estimates[0], estimates[1])

    def test_expert_parallelism_moves_whole_experts_without_shrinking_weights(self):
        model = self.hardware.model("Qwen3-235B-A22B")
        sizes = []
        for ep in (False, True):
            parallel = ParallelConfig(tensor_parallel_size=8, enable_expert_parallel=ep)
            cache = CacheConfig(
                16,
                self.hardware.device("b200"),
                model,
                parallel,
                expert_placement=build_expert_placement(
                    model.moe, parallel, model.num_layers
                ),
            )
            sizes.append(cache.model_param_size)
        # 128 experts as TP8 shards or as 16 whole experts per rank.
        self.assertEqual(sizes[0], sizes[1])

    def test_generic_moe_prices_shared_experts(self):
        source = self.hardware.model("Qwen3-235B-A22B")
        shared = replace(
            source,
            model_id="shared-expert-moe",
            aliases=(),
            moe=replace(source.moe, num_shared_experts=1),
        )
        package = self.hardware.operator_package(
            self.hardware.device("b200").device_id, "vllm"
        )
        latencies = []
        for model in (source, shared):
            backend = OperatorTableLatencyBackend(
                device=self.hardware.device("b200"),
                model=model,
                parallel_config=ParallelConfig(tensor_parallel_size=8),
                package=package,
            )
            latencies.append(backend._moe_layers(64, []))
        self.assertGreater(backend.shared_experts(64, tp=8), 0)
        self.assertAlmostEqual(
            latencies[1] - latencies[0],
            backend.moe_layers_local * backend.shared_experts(64, tp=8),
        )

    def test_unanswered_queries_are_recorded_once_unless_analytical_only(self):
        model = self.hardware.model("Qwen3-235B-A22B")
        for fallback, expected in (("table_first", 2), ("analytical_only", 0)):
            backend = OperatorTableLatencyBackend(
                device=self.hardware.device("b200"),
                model=model,
                parallel_config=ParallelConfig(tensor_parallel_size=8),
                fallback=fallback,
            )
            for _ in range(3):
                backend.gemm(64, 512, 4096)
                backend._moe_layers(64, [])
            missing = backend.stats.missing
            self.assertEqual(missing.query_count, expected)
            self.assertEqual(
                {group.table for group in missing.groups.values()},
                {"gemm", "moe"} if expected else set(),
            )

    def test_compression_and_forward_ownership_follow_configuration(self):
        source = self.hardware.model("DeepSeek-V4-Pro")
        geometry = replace(
            source.components,
            indexed_compress_ratio=8,
            compress_ratios=tuple(
                8 if r == 4 else 256 if r == 128 else r
                for r in source.components.compress_ratios
            ),
        )
        model = replace(source, model_id="custom-compression", components=geometry)
        layout = hybrid_kv_layout(model)
        self.assertTrue(
            any(
                page.tokens_per_state == 8
                for group in layout.groups
                for page in group.layers
            )
        )
        ranks = {0: RankShape(8, 1, False, 8, 7, 7)}
        step = ForwardPass(ranks)
        self.assertIs(step.derived(ranks), step.derived(dict(ranks)))
        self.assertEqual(sum(sp_valid_rows(7, 8, 4, rank) for rank in range(4)), 7)
