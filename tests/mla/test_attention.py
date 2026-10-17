from dataclasses import replace
import json
from tempfile import TemporaryDirectory
import unittest

from TokenSim.af.providers.operator_table import OperatorTableAFComputeProvider
from TokenSim.af.providers.queries import AFComputeQuery
from TokenSim.af.runtime.topology import kv_head_partitions
from TokenSim.errors import ConfigurationError
from TokenSim.operator_data.analytical import Calibration, analytical_model_for
from TokenSim.operator_data.lookup import (
    LookupPolicy,
    MissingOperatorDataError,
    OperatorLookup,
)
from TokenSim.operator_data.package import (
    OperatorDataPackage,
    PackageMeta,
    SourceRecord,
)
from TokenSim.operator_data.workload import work_for_table
from scripts.deepseek_v3_runtime import ROOT, configuration, simulate


def core_key(**overrides):
    key = dict(
        attn_dtype="bf16",
        kv_cache_dtype="bf16",
        phase="decode",
        execution_mode="absorbed",
        batch_size=1,
        query_len=1,
        context_lengths="[[16384,1]]",
        num_heads=128,
        kv_lora_rank=512,
        rope_head_dim=64,
        nope_head_dim=128,
        v_head_dim=128,
    )
    key.update(overrides)
    return key


def query(shape, contexts=(16, 24), phase="decode", tp=1):
    tokens = len(contexts) if phase == "decode" else sum(contexts)
    local = dict(
        tokens=tokens,
        hidden_size=shape.hidden_size,
        global_hidden_size=shape.hidden_size,
        **shape.mla_dimensions.local_shape(shape.num_attention_heads, tp),
    )
    return AFComputeQuery(
        iteration_id=0,
        block_id=0,
        layer_id=0,
        phase=phase,
        stage="attention",
        token_count=tokens,
        context_lengths=contexts,
        model_id=shape.model_id,
        dtype="bf16",
        device_type="a100_sxm_80g",
        worker_id=0,
        local_token_count=tokens,
        local_shape=local,
        layer_type="moe",
        original_token_count=tokens,
    )


class MLAFormulaTest(unittest.TestCase):
    def test_decode_cache_is_one_latent_vector_not_per_head_k_and_v(self):
        result = work_for_table("mla_attention", core_key())
        self.assertEqual(result.details["cache_read_bytes"], 18 * 1024**2)
        self.assertEqual(result.details["cache_write_bytes"], 1152)
        self.assertEqual(result.kv_bytes, 18 * 1024**2 + 1152)
        self.assertEqual(result.flops, 2 * 128 * 16384 * (1024 + 64) + 5 * 128 * 16384)
        half = work_for_table("mla_attention", core_key(num_heads=64))
        self.assertEqual(half.kv_bytes, result.kv_bytes)
        self.assertEqual(half.flops * 2, result.flops)

    def test_ragged_and_fp8_cache_work(self):
        batch = work_for_table(
            "mla_attention",
            core_key(batch_size=2, context_lengths="[[10,0.5],[20,0.5]]"),
        )
        singles = [
            work_for_table("mla_attention", core_key(context_lengths=f"[[{n},1]]"))
            for n in (10, 20)
        ]
        self.assertEqual(batch.flops, sum(x.flops for x in singles))
        self.assertEqual(batch.kv_bytes, sum(x.kv_bytes for x in singles))
        fp8 = work_for_table("mla_attention", core_key(kv_cache_dtype="fp8"))
        bf16 = work_for_table("mla_attention", core_key())
        self.assertEqual(fp8.kv_bytes * 2, bf16.kv_bytes)
        self.assertEqual(fp8.flops, bf16.flops)

    def test_prefill_expanded_causal_core_and_cached_formula(self):
        key = core_key(
            phase="prefill",
            execution_mode="expanded",
            batch_size=2,
            query_len=8,
            context_lengths="[[8,1]]",
        )
        result = work_for_table("mla_attention", key)
        self.assertEqual(
            result.flops, 2 * 128 * (2 * 36) * (128 + 64 + 128) + 5 * 128 * 2 * 36
        )
        self.assertEqual(result.details["cache_read_bytes"], 0)
        self.assertEqual(result.kv_bytes, 16 * 576 * 2)
        # Analytical work supports cached queries; standard executor rejects them separately.
        cached = work_for_table("mla_attention", dict(key, context_lengths="[[16,1]]"))
        self.assertEqual(cached.details["cache_read_bytes"], 2 * 8 * 576 * 2)
        for change in (
            dict(execution_mode="expanded"),
            dict(query_len=2),
            dict(phase="unknown"),
        ):
            with self.assertRaises(ValueError):
                work_for_table("mla_attention", core_key(**change))

    def test_grouped_gemm_has_head_local_weights_and_one_launch(self):
        config = configuration()
        device = config.hardware.device("a100_sxm_80g")
        key = dict(dtype="bf16", m=2, n=512, k=128, groups=128)
        result = analytical_model_for(device.family).estimate(
            "batched_gemm", key, device, Calibration(offset_us=3)
        )
        self.assertEqual(result.work.flops, 2 * 2 * 512 * 128 * 128)
        self.assertEqual(result.work.weight_bytes, 512 * 128 * 128 * 2)
        self.assertEqual(
            result.overhead_us, device.analytical.get("kernel_launch_us", 4)
        )
        self.assertAlmostEqual(result.latency_us, result.roofline_us + 3)


class MLAIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = configuration()
        cls.shape = cls.config.model

    def test_metadata_matches_official_v3_config_and_validates(self):
        shape = self.shape
        reference = json.loads(
            (ROOT / "data/models/deepseek-v3-hf-config.json").read_text()
        )
        for local, official in (
            ("q_lora_rank", "q_lora_rank"),
            ("kv_lora_rank", "kv_lora_rank"),
            ("rope_head_dim", "qk_rope_head_dim"),
            ("nope_head_dim", "qk_nope_head_dim"),
            ("v_head_dim", "v_head_dim"),
        ):
            self.assertEqual(getattr(shape, local), reference[official])
        self.assertEqual(shape.head_dim, 192)
        self.assertEqual(shape.mla_dimensions.cache_width, 576)
        self.assertEqual(
            replace(shape, q_lora_rank=1024).mla_dimensions.cache_width, 576
        )
        for field in (
            "q_lora_rank",
            "kv_lora_rank",
            "rope_head_dim",
            "nope_head_dim",
            "v_head_dim",
        ):
            for value in (0, -1, 100000, True):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ConfigurationError, field),
                ):
                    replace(shape, **{field: value})

    def test_provider_decomposes_actual_mla_projection_dimensions(self):
        provider = OperatorTableAFComputeProvider(
            self.config.hardware, self.shape, fallback="analytical_only"
        )
        result = provider.estimate(query(self.shape))
        components = {c["name"]: c for c in result.details["components"]}
        self.assertNotIn("qkv_projection", components)
        self.assertEqual(components["mla_q_up"]["key"]["n"], 128 * 192)
        self.assertEqual(components["mla_q_absorb"]["key"]["groups"], 128)
        self.assertEqual(components["mla_q_absorb"]["key"]["n"], 512)
        self.assertEqual(components["output_projection"]["key"]["k"], 128 * 128)
        self.assertEqual(
            components["mla_attention_core"]["key"]["context_lengths"],
            "[[16,0.5],[24,0.5]]",
        )
        self.assertNotIn("q_lora_rank", components["mla_attention_core"]["key"])
        prefill = provider.estimate(query(self.shape, (8, 8), "prefill"))
        names = {c["name"] for c in prefill.details["components"]}
        self.assertIn("mla_kv_expand", names)
        self.assertNotIn("mla_q_absorb", names)
        self.assertNotIn("mla_value_expand", names)
        chunked = provider.estimate(
            replace(
                query(self.shape, (16, 24), "prefill"),
                token_count=8,
                local_token_count=8,
                original_token_count=8,
                local_shape=dict(query(self.shape).local_shape, tokens=8),
            )
        )
        chunked_core = next(
            c
            for c in chunked.details["components"]
            if c["name"] == "mla_attention_core"
        )
        self.assertEqual(chunked_core["key"]["query_len"], 4)
        self.assertEqual(chunked_core["key"]["context_lengths"], "[[16,0.5],[24,0.5]]")
        with self.assertRaisesRegex(ConfigurationError, "positive context lengths"):
            provider.estimate(replace(query(self.shape, (16,)), context_lengths=()))

    def test_tp_cache_replicates_and_weights_shard(self):
        plan = configuration(a_tp=2).plan
        partitions = kv_head_partitions(
            num_kv_heads=1,
            tensor_parallel_size=2,
            head_dim=192,
            num_layers=61,
            block_size=16,
            dtype_size_bytes=2,
            mla_cache_width=576,
        )
        for partition in partitions:
            self.assertEqual(partition.bytes_per_token, 576 * 2 * 61)
            self.assertEqual(partition.bytes_per_block, 576 * 2 * 61 * 16)
            self.assertEqual(partition.replica_factor, 2)
        for binding in plan.rank_bindings:
            if binding.capability == "af_attention":
                self.assertEqual(binding.kv_partition.bytes_per_token, 576 * 2)
        expected = self.shape.mla_dimensions.weight_elements(64, 7168) * 2
        attention = [
            c
            for p in plan.physical_states
            for c in p.components
            if c.category == "attention"
        ]
        self.assertEqual(len(attention), 2)
        self.assertTrue(all(c.bytes == expected for c in attention))
        embeddings = [
            c
            for p in plan.physical_states
            for c in p.components
            if c.category == "embedding_lm_head"
        ]
        self.assertEqual(sum(c.bytes for c in embeddings), 129280 * 7168 * 2 * 2)

    def test_full_model_capacity_and_dense_layer_layout(self):
        plan = configuration(full_model=True).plan
        self.assertEqual(len(plan.layers), 61)
        self.assertEqual(
            [p.layer_type for p in plan.layers[:4]], ["dense", "dense", "dense", "moe"]
        )
        self.assertEqual(sum(p.layer_type == "moe" for p in plan.layers), 58)
        self.assertTrue(
            all(p.model_bytes < p.capacity_bytes for p in plan.physical_states)
        )
        self.assertEqual(len({p.capacity_bytes for p in plan.physical_states}), 1)
        self.assertLess(
            plan.physical_states[0].capacity_bytes,
            self.config.hardware.device("a100_sxm_80g").memory_capacity_bytes.value,
        )
        self.assertGreater(sum(p.model_bytes for p in plan.physical_states), 1.3e12)
        with self.assertRaisesRegex(Exception, "capacity"):
            configuration(full_model=True, f_tp=2, a_tp=2)

    def test_tp_two_runs_real_decode_with_local_heads_and_blocks(self):
        executor, rows, _, _ = simulate(
            configuration(a_tp=2, af_batch_size=1), 2, 16, fallback="analytical_only"
        )
        cores = [
            c["key"]
            for e in executor.trace.events
            for c in e.details.get("components", ())
            if c["name"] == "mla_attention_core"
        ]
        self.assertTrue(cores)
        self.assertTrue(
            all(k["num_heads"] == 64 and k["batch_size"] == 1 for k in cores)
        )
        self.assertGreater(rows[0]["makespan_us"], 0)
        self.assertTrue(
            any(e.event_type == "collective" for e in executor.trace.events)
        )
        stages = {e.stage for e in executor.trace.events}
        self.assertTrue(
            {
                "attention",
                "routed_expert",
                "shared_expert",
                "weighted_combine",
                "residual",
            }
            <= stages
        )
        executor.trace.validate_invariants(0)

    def test_offline_package_roundtrip_missing_shape_and_measured_override(self):
        key = core_key()
        device = self.config.hardware.device("a100_sxm_80g")
        estimated = analytical_model_for(device.family).estimate(
            "mla_attention", key, device
        )
        for source, latency in (
            (
                SourceRecord("analytical-fixture", "D", "analytical"),
                estimated.latency_us,
            ),
            (SourceRecord("measured-fixture", "A", "measured"), 123.0),
        ):
            package = OperatorDataPackage.from_rows(
                PackageMeta(
                    "fixture-v1",
                    device.device_id,
                    "test",
                    sources={source.source_id: source},
                ),
                {
                    "mla_attention": [
                        dict(key, latency_us=latency, source_id=source.source_id)
                    ]
                },
            )
            with TemporaryDirectory() as directory:
                package.write(directory)
                lookup = OperatorLookup(
                    OperatorDataPackage.load(directory),
                    LookupPolicy(extrapolate="none"),
                )
                result = lookup.lookup("mla_attention", key)
                self.assertEqual(
                    (result.latency_us, result.source_id, result.match_type),
                    (latency, source.source_id, "exact"),
                )
                for mutation in (
                    dict(kv_lora_rank=256),
                    dict(context_lengths="[[17,1]]"),
                    dict(v_head_dim=64),
                ):
                    with self.assertRaises(MissingOperatorDataError):
                        lookup.lookup("mla_attention", dict(key, **mutation))


if __name__ == "__main__":
    unittest.main()
